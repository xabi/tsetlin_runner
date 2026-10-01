defmodule TsetlinRunnerTest do
  use ExUnit.Case, async: true

  describe "radius_for_clause_size/1" do
    test "inverts ground_feature_len for known radii" do
      # (2r+1)^2 * 5 + 2, tsetlin_world/src/GroundTM.jl's ground_feature_len/1
      assert TsetlinRunner.radius_for_clause_size(47) == {:ok, 1}
      assert TsetlinRunner.radius_for_clause_size(407) == {:ok, 4}
      assert TsetlinRunner.radius_for_clause_size(1447) == {:ok, 8}
      assert TsetlinRunner.radius_for_clause_size(7) == {:ok, 0}
    end

    test "returns :error for a clause_size that isn't a GroundTM radius encoding" do
      assert TsetlinRunner.radius_for_clause_size(2) == :error
      assert TsetlinRunner.radius_for_clause_size(48) == :error
      assert TsetlinRunner.radius_for_clause_size(0) == :error
      assert TsetlinRunner.radius_for_clause_size(1) == :error
    end

    test "round-trips every radius from 0 to 32" do
      for r <- 0..32 do
        clause_size = (2 * r + 1) * (2 * r + 1) * 5 + 2
        assert TsetlinRunner.radius_for_clause_size(clause_size) == {:ok, r}
      end
    end
  end

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

      try do
        assert TsetlinRunner.load(path) == {:error, :invalid_format}
      after
        File.rm(path)
      end
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

    test "clause_size/1 returns the input bit length the model was trained with" do
      with_tiny_model(fn model ->
        assert TsetlinRunner.clause_size(model) == 2
      end)
    end

    test "predict/2 returns a bit_length_mismatch tuple for the wrong input size" do
      with_tiny_model(fn model ->
        too_short = <<0::little-32>>
        assert TsetlinRunner.predict(model, too_short) == {:error, :bit_length_mismatch}
      end)
    end

    test "load/1 returns an invalid_format tuple for a header with semantically invalid fields" do
      # Same layout as tiny_model_bytes/0, but classes_num is corrupted to 0,
      # which is invalid for kind=General (classes_num must be non-zero) --
      # this used to parse "successfully" and later panic in predict/2's
      # out-of-bounds `classes[0]`/`classes[1]` access instead of surfacing
      # as a tagged error.
      path =
        Path.join(
          System.tmp_dir!(),
          "tsetlin_runner_invalid_header_#{System.unique_integer([:positive])}.tmbin"
        )

      invalid_bytes = <<
        "TSTM",
        1::little-32,
        1::8,
        2::little-32,
        1::little-32,
        0::little-32,
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

      File.write!(path, invalid_bytes)

      try do
        assert TsetlinRunner.load(path) == {:error, :invalid_format}
      after
        File.rm(path)
      end
    end
  end

  describe "classify_frame/4" do
    # A 7-bit model (radius=0: (2*0+1)^2*5+2 = 7), one clause per polarity,
    # so it always predicts class 1 regardless of input -- this task only
    # needs a structurally valid model, not a semantically meaningful one
    # (numeric correctness against Julia is a separate task).
    defp tiny_7bit_model_bytes do
      <<
        "TSTM",
        1::little-32,
        0::8,
        7::little-32,
        1::little-32,
        2::little-32,
        1::little-32,
        1::little-64,
        1::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        0::little-64
      >>
    end

    defp with_tiny_7bit_model(fun) do
      path =
        Path.join(System.tmp_dir!(), "tsetlin_runner_7bit_#{System.unique_integer([:positive])}.tmbin")

      File.write!(path, tiny_7bit_model_bytes())

      try do
        {:ok, model} = TsetlinRunner.load(path)
        fun.(model)
      after
        File.rm(path)
      end
    end

    @fixture_jpeg File.read!(
                     Path.join([
                       __DIR__,
                       "..",
                       "native/tsetlin_nif/tests/fixtures/tiny_solid.jpg"
                     ])
                   )

    test "returns a fully-classified out_w*out_h grid for a valid frame" do
      with_tiny_7bit_model(fn model ->
        assert {:ok, grid} = TsetlinRunner.classify_frame(model, @fixture_jpeg, 2, 2, 0)
        assert length(grid) == 4
        assert Enum.all?(grid, &(&1 in [0, 1]))
      end)
    end

    test "returns invalid_jpeg for malformed frame bytes" do
      with_tiny_7bit_model(fn model ->
        assert TsetlinRunner.classify_frame(model, <<0, 1, 2, 3>>, 2, 2, 0) ==
                 {:error, :invalid_jpeg}
      end)
    end

    test "returns invalid_dimensions for out_w=0 without attempting to decode" do
      with_tiny_7bit_model(fn model ->
        # Deliberately-invalid JPEG bytes: if dimensions were checked AFTER
        # decode, this would fail with :invalid_jpeg instead.
        assert TsetlinRunner.classify_frame(model, <<0, 1, 2, 3>>, 0, 2, 0) ==
                 {:error, :invalid_dimensions}
      end)
    end

    test "returns bit_length_mismatch when radius doesn't match the model's clause_size" do
      with_tiny_7bit_model(fn model ->
        # This model's clause_size is 7 (radius=0). radius=1 -> 47 bits.
        assert TsetlinRunner.classify_frame(model, @fixture_jpeg, 2, 2, 1) ==
                 {:error, :bit_length_mismatch}
      end)
    end

    test "returns invalid_dimensions for an oversized radius, without decoding" do
      with_tiny_7bit_model(fn model ->
        # Final review (2026-09-28): (2*radius+1)^2*5+2 computed in u32
        # wraps for a large radius, so an unchecked radius could sail past
        # bit_length_mismatch and then loop ~2^64 times inside cell_bits.
        # Garbage JPEG bytes here prove the check runs before any decode
        # attempt (an :invalid_jpeg result would mean it decoded first).
        assert TsetlinRunner.classify_frame(model, <<0, 1, 2, 3>>, 2, 2, 100_000) ==
                 {:error, :invalid_dimensions}
      end)
    end

    test "returns invalid_dimensions for an oversized out_w/out_h, without decoding" do
      with_tiny_7bit_model(fn model ->
        assert TsetlinRunner.classify_frame(model, <<0, 1, 2, 3>>, 100_000, 2, 0) ==
                 {:error, :invalid_dimensions}
      end)
    end
  end

  describe "classify_frame/4 matches the Julia ground_tm pipeline" do
    # Final review (2026-09-28): the original version of this test only
    # spot-checked two cells and asserted `sky != grass` -- it never called
    # the real `classify_frame/4` output against the Julia-computed labels
    # this fixture already carries, so a bug specific to classify_frame_nif's
    # own wiring (as opposed to the lower-level pipeline pieces the Rust
    # parity test exercises directly) could ship undetected. Compare the
    # full 768-cell grid instead -- the data was already being generated
    # and checked in; comparing all of it costs nothing extra.
    test "matches every one of the Julia-computed labels" do
      jpeg =
        File.read!(
          Path.join([__DIR__, "..", "native/tsetlin_nif/tests/fixtures/ground_tm_parity.jpg"])
        )

      model_path =
        Path.join([__DIR__, "..", "native/tsetlin_nif/tests/fixtures/ground_tm_parity.tmbin"])

      expected_path =
        Path.join([
          __DIR__,
          "..",
          "native/tsetlin_nif/tests/fixtures/ground_tm_parity_expected.txt"
        ])

      {:ok, model} = TsetlinRunner.load(model_path)

      [header | rows] = expected_path |> File.read!() |> String.split("\n", trim: true)
      [out_w, out_h, radius] = header |> String.split(" ") |> Enum.map(&String.to_integer/1)

      expected =
        Map.new(rows, fn row ->
          [col, row_idx, label] = row |> String.split(" ") |> Enum.map(&String.to_integer/1)
          {{col, row_idx}, label}
        end)

      assert {:ok, grid} = TsetlinRunner.classify_frame(model, jpeg, out_w, out_h, radius)
      assert length(grid) == out_w * out_h

      mismatches =
        for row <- 0..(out_h - 1), col <- 0..(out_w - 1) do
          actual = Enum.at(grid, row * out_w + col)
          expected_label = Map.fetch!(expected, {col, row})
          if actual != expected_label, do: {col, row, expected_label, actual}
        end
        |> Enum.reject(&is_nil/1)

      assert mismatches == []
    end
  end
end
