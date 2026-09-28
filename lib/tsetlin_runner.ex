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
end
