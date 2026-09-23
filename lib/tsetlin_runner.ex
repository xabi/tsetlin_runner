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
end
