defmodule TsetlinRunner.TargetTest do
  use ExUnit.Case, async: false

  describe "resolve/1" do
    test "maps known Nerves rpi targets to their Rust triples" do
      assert TsetlinRunner.Target.resolve("rpi0") == {:ok, "arm-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi") == {:ok, "arm-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi2") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi3") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi3a") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi4") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("bbb") == {:ok, "armv7-unknown-linux-gnueabihf"}
    end

    test "treats nil and \"host\" as no cross-compilation" do
      assert TsetlinRunner.Target.resolve(nil) == {:ok, :native}
      assert TsetlinRunner.Target.resolve("host") == {:ok, :native}
    end

    test "reports an unknown target as an error instead of guessing" do
      assert TsetlinRunner.Target.resolve("some_future_board") == {:error, :unknown_target}
    end
  end

  describe "configure_cargo_target!/0" do
    setup do
      original = System.get_env("MIX_TARGET")
      original_cargo = System.get_env("CARGO_BUILD_TARGET")

      on_exit(fn ->
        if original, do: System.put_env("MIX_TARGET", original), else: System.delete_env("MIX_TARGET")
        if original_cargo, do: System.put_env("CARGO_BUILD_TARGET", original_cargo), else: System.delete_env("CARGO_BUILD_TARGET")
      end)

      :ok
    end

    test "leaves CARGO_BUILD_TARGET unset when MIX_TARGET is unset" do
      System.delete_env("MIX_TARGET")
      System.delete_env("CARGO_BUILD_TARGET")

      assert TsetlinRunner.Target.configure_cargo_target!() == :ok
      assert System.get_env("CARGO_BUILD_TARGET") == nil
    end

    test "sets CARGO_BUILD_TARGET for a known Nerves target" do
      System.put_env("MIX_TARGET", "rpi0")

      assert TsetlinRunner.Target.configure_cargo_target!() == :ok
      assert System.get_env("CARGO_BUILD_TARGET") == "arm-unknown-linux-gnueabihf"
    end

    test "raises for an unknown MIX_TARGET instead of silently building for the host" do
      System.put_env("MIX_TARGET", "some_future_board")

      assert_raise RuntimeError, ~r/unknown MIX_TARGET/, fn ->
        TsetlinRunner.Target.configure_cargo_target!()
      end
    end
  end
end
