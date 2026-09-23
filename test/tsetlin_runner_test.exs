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

  describe "load/1 and predict/2 (end-to-end against a tiny fixture model)" do
    # Mirrors native/tsetlin_nif/src/format.rs's tiny_general_model_bytes():
    # 2-bit input, 2 classes, lf=2, one clause per polarity per class.
    # Class 1 requires both input bits true; class 2 requires both false.
    defp tiny_model_bytes do
      <<
        "TSTM",
        1::little-32,
        1::8,
        2::little-32,
        1::little-32,
        2::little-32,
        1::little-32,
        2::little-64,
        1::little-64,
        2::little-64,
        3::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        3::little-64,
        0::little-64,
        0::little-64
      >>
    end

    defp with_tiny_model(fun) do
      path = Path.join(System.tmp_dir!(), "tsetlin_runner_test_#{System.unique_integer([:positive])}.tmbin")
      File.write!(path, tiny_model_bytes())

      try do
        {:ok, model} = TsetlinRunner.load(path)
        fun.(model)
      after
        File.rm(path)
      end
    end

    test "load/1 returns an io_error tuple for a missing file" do
      assert TsetlinRunner.load("/nonexistent/path.tmbin") == {:error, :io_error}
    end

    test "load/1 returns an invalid_format tuple for garbage bytes" do
      path = Path.join(System.tmp_dir!(), "tsetlin_runner_garbage_#{System.unique_integer([:positive])}.tmbin")
      File.write!(path, <<0, 1, 2, 3>>)

      assert TsetlinRunner.load(path) == {:error, :invalid_format}
      File.rm(path)
    end

    test "predicts class 2 when both input bits are false" do
      with_tiny_model(fn model ->
        bits = TsetlinRunner.pack_bits([false, false])
        assert TsetlinRunner.predict(model, bits) == {:ok, 2}
      end)
    end

    test "predicts class 1 when both input bits are true" do
      with_tiny_model(fn model ->
        bits = TsetlinRunner.pack_bits([true, true])
        assert TsetlinRunner.predict(model, bits) == {:ok, 1}
      end)
    end

    test "breaks a vote tie in favor of the first class" do
      with_tiny_model(fn model ->
        assert TsetlinRunner.predict(model, TsetlinRunner.pack_bits([true, false])) == {:ok, 1}
        assert TsetlinRunner.predict(model, TsetlinRunner.pack_bits([false, true])) == {:ok, 1}
      end)
    end

    test "predict/2 returns a bit_length_mismatch tuple for the wrong input size" do
      with_tiny_model(fn model ->
        too_short = <<0::little-32>>
        assert TsetlinRunner.predict(model, too_short) == {:error, :bit_length_mismatch}
      end)
    end
  end
end
