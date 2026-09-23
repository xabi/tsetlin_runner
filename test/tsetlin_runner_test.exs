defmodule TsetlinRunnerTest do
  use ExUnit.Case, async: true

  describe "pack_bits/1" do
    test "packs an empty list into an empty binary" do
      assert TsetlinRunner.pack_bits([]) == <<>>
    end

    test "packs fewer than 64 bits into a single little-endian u64, LSB first" do
      assert TsetlinRunner.pack_bits([true, true]) == <<3::little-64>>
      assert TsetlinRunner.pack_bits([true, false]) == <<1::little-64>>
      assert TsetlinRunner.pack_bits([false, true]) == <<2::little-64>>
      assert TsetlinRunner.pack_bits([false, false]) == <<0::little-64>>
    end

    test "spans multiple 64-bit chunks without losing bit 64" do
      # 64 `false` bits (chunk 0 == 0), then a single `true` bit, which
      # must land as bit 0 of chunk 1 (index 64 overall) -- a naive
      # chunking bug would drop it or shift it into the wrong chunk.
      bits = List.duplicate(false, 64) ++ [true]

      assert TsetlinRunner.pack_bits(bits) == <<0::little-64, 1::little-64>>
    end
  end

  describe "TsetlinRunner.Native (raw NIF bindings)" do
    test "load_model_nif/1 returns an io_error tuple for a missing file" do
      assert TsetlinRunner.Native.load_model_nif("/nonexistent/path.tmbin") ==
               {:error, :io_error}
    end
  end
end
