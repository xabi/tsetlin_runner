defmodule TsetlinRunner do
  @moduledoc """
  Runs inference for Tsetlin Machine classifiers exported from the Julia
  `tsetlin_world` project.
  """

  import Bitwise

  alias TsetlinRunner.Native

  @doc """
  Packs a list of booleans into the little-endian, LSB-first, 64-bit-chunk
  layout Tsetlin Machine models expect (matching `Tsetlin.jl`'s `TMInput`
  bit layout). Bit `i` of the input (0-indexed) becomes bit `rem(i, 64)`
  of chunk `div(i, 64)`.
  """
  @spec pack_bits([boolean()]) :: binary()
  def pack_bits(bits) when is_list(bits) do
    bits
    |> Enum.chunk_every(64)
    |> Enum.map(&pack_chunk/1)
    |> IO.iodata_to_binary()
  end

  defp pack_chunk(chunk_bits) do
    value =
      chunk_bits
      |> Enum.with_index()
      |> Enum.reduce(0, fn
        {true, index}, acc -> acc ||| 1 <<< index
        {false, _index}, acc -> acc
      end)

    <<value::unsigned-little-64>>
  end

  @doc """
  Loads a compiled Tsetlin Machine model (a `.tmbin` file produced by the
  Julia `tsetlin_world` project's exporter) from `path`.
  """
  @spec load(String.t()) :: {:ok, reference()} | {:error, :invalid_format | :io_error}
  def load(path) when is_binary(path) do
    Native.load_model_nif(path)
  end

  @doc """
  Runs inference for `packed_bits` (see `pack_bits/1`) against a `model`
  returned by `load/1`.
  """
  @spec predict(reference(), binary()) :: {:ok, integer()} | {:error, :bit_length_mismatch}
  def predict(model, packed_bits) when is_reference(model) and is_binary(packed_bits) do
    Native.predict_nif(model, packed_bits)
  end

  @doc """
  Classifies every cell of an `out_w`x`out_h` grid from a raw JPEG frame in
  one call: JPEG decode, box-average resize, feature-map computation, and
  per-cell `predict/2` -- replicating `tsetlin_world`'s Julia pipeline
  exactly (see docs/superpowers/specs/2026-09-28-ground-vision-nif-design.md).

  Returns a flat, row-major list (`index = row * out_w + col`, 0-indexed)
  of the loaded model's own class labels -- this is a different order
  than `tsetlin_world`'s column-major Julia grids.

  `radius` must match whatever radius the loaded model was trained with
  (i.e. `ground_feature_len(radius)` must equal the model's `clause_size`),
  or this returns `{:error, :bit_length_mismatch}`.

  `out_w`/`out_h` of `0` are accepted at the Elixir boundary (rather than
  raising `FunctionClauseError`) so the NIF's own fail-fast dimension
  check can return the documented `{:error, :invalid_dimensions}` instead.
  """
  @spec classify_frame(reference(), binary(), non_neg_integer(), non_neg_integer(), non_neg_integer()) ::
          {:ok, [integer()]} | {:error, atom()}
  def classify_frame(model, jpeg, out_w, out_h, radius)
      when is_reference(model) and is_binary(jpeg) and
             is_integer(out_w) and out_w >= 0 and
             is_integer(out_h) and out_h >= 0 and
             is_integer(radius) and radius >= 0 do
    Native.classify_frame_nif(model, jpeg, out_w, out_h, radius)
  end

  @doc """
  The input bit length (`clause_size`) `model` was compiled with -- the same
  field `classify_frame/5` already validates its own `radius` argument
  against internally (returning `{:error, :bit_length_mismatch}` on a
  mismatch). Exposed so a caller can derive the right `radius` for a loaded
  model up front via `radius_for_clause_size/1`, instead of hardcoding one
  and risking it drifting out of sync with whatever the model was actually
  trained with.
  """
  @spec clause_size(reference()) :: non_neg_integer()
  def clause_size(model) when is_reference(model), do: Native.clause_size_nif(model)

  @doc """
  Inverts `ground_feature_len(radius) = (2*radius+1)^2 * 5 + 2`
  (`tsetlin_world/src/GroundTM.jl`) -- the GroundTM feature encoding this
  library's `classify_frame/5` replicates. Returns `{:ok, radius}` for a
  `clause_size` that is an exact match for some non-negative integer
  radius, `:error` otherwise (not a GroundTM-shaped model, or a corrupt/
  unrelated `.tmbin`).
  """
  @spec radius_for_clause_size(non_neg_integer()) :: {:ok, non_neg_integer()} | :error
  def radius_for_clause_size(clause_size) when is_integer(clause_size) and clause_size >= 2 do
    # (2r+1)^2 = (clause_size - 2) / 5
    with 0 <- rem(clause_size - 2, 5),
         square when square >= 1 <- div(clause_size - 2, 5),
         root <- isqrt(square),
         true <- root * root == square,
         0 <- rem(root - 1, 2) do
      {:ok, div(root - 1, 2)}
    else
      _ -> :error
    end
  end

  def radius_for_clause_size(_clause_size), do: :error

  defp isqrt(n) when is_integer(n) and n >= 0, do: n |> :math.sqrt() |> round() |> newton_fix(n)

  # :math.sqrt/1's float round-trip can be off by one for large perfect
  # squares -- nudge to the exact integer root before the caller's
  # root*root == square check, rather than trusting the float directly.
  defp newton_fix(guess, n) when guess * guess > n, do: newton_fix(guess - 1, n)
  defp newton_fix(guess, n) when (guess + 1) * (guess + 1) <= n, do: newton_fix(guess + 1, n)
  defp newton_fix(guess, _n), do: guess
end
