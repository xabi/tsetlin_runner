defmodule TsetlinRunner do
  @moduledoc """
  Runs inference for Tsetlin Machine classifiers exported from the Julia
  `tsetlin_world` project.
  """

  import Bitwise

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
end
