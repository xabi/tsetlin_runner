defmodule TsetlinRunner.Target do
  @moduledoc """
  Maps a Nerves `MIX_TARGET` value to the Rust target triple Cargo should
  cross-compile the `tsetlin_nif` crate for. Defined here (rather than
  under `lib/`) because `mix.exs` needs it before anything in `lib/` has
  been compiled.
  """

  @mapping %{
    "rpi0" => "arm-unknown-linux-gnueabihf",
    "rpi" => "arm-unknown-linux-gnueabihf",
    "rpi2" => "armv7-unknown-linux-gnueabihf",
    "rpi3" => "armv7-unknown-linux-gnueabihf",
    "rpi3a" => "armv7-unknown-linux-gnueabihf",
    "rpi4" => "armv7-unknown-linux-gnueabihf",
    "bbb" => "armv7-unknown-linux-gnueabihf"
  }

  @spec resolve(String.t() | nil) :: {:ok, :native} | {:ok, String.t()} | {:error, :unknown_target}
  def resolve(nil), do: {:ok, :native}
  def resolve("host"), do: {:ok, :native}

  def resolve(mix_target) when is_binary(mix_target) do
    case Map.fetch(@mapping, mix_target) do
      {:ok, triple} -> {:ok, triple}
      :error -> {:error, :unknown_target}
    end
  end

  @doc """
  Reads `MIX_TARGET` from the OS environment and sets `CARGO_BUILD_TARGET`
  accordingly, so Rustler's `cargo build` cross-compiles automatically.
  Raises on an unrecognized `MIX_TARGET` rather than silently falling
  back to a host build.

  Must be called at the call site that actually triggers Rustler's
  compile (`TsetlinRunner.Native`, right before `use Rustler`), not from
  this file's top level: under Nerves, `$CC` is exported by
  `Nerves.Env.bootstrap/0` (run via `mix nerves.loadpaths`), which fires
  after Mix has already loaded a `path:`-referenced dependency's
  `mix.exs` to resolve the dependency graph. Calling this here would read
  `$CC` before Nerves sets it and silently skip the linker override.
  """
  @spec configure_cargo_target!() :: :ok
  def configure_cargo_target! do
    case resolve(System.get_env("MIX_TARGET")) do
      {:ok, :native} ->
        :ok

      {:ok, triple} ->
        System.put_env("CARGO_BUILD_TARGET", triple)

        linker_var =
          "CARGO_TARGET_" <> String.upcase(String.replace(triple, "-", "_")) <> "_LINKER"

        case System.get_env("CC") do
          nil ->
            if System.get_env(linker_var) == nil do
              raise "tsetlin_runner: cross-compiling for #{triple} but $CC is unset and " <>
                      "$#{linker_var} is not already set -- Cargo would silently fall back " <>
                      "to the host linker and fail to link ARM objects. Set $CC (Nerves does " <>
                      "this via `mix nerves.loadpaths`) or set $#{linker_var} directly."
            end

          cc ->
            System.put_env(linker_var, cc)
        end

        :ok

      {:error, :unknown_target} ->
        raise "tsetlin_runner: unknown MIX_TARGET=#{inspect(System.get_env("MIX_TARGET"))}, " <>
                "add it to TsetlinRunner.Target's mapping table"
    end
  end
end

defmodule TsetlinRunner.MixProject do
  use Mix.Project

  def project do
    [
      app: :tsetlin_runner,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:rustler, "~> 0.38.0"}
    ]
  end
end
