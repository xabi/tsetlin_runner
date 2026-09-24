# tsetlin_runner

An Elixir library that runs **inference** for [Tsetlin Machine][tsetlin-jl]
classifiers trained with [`BooBSD/Tsetlin.jl`][tsetlin-jl], via a Rust NIF
([Rustler][rustler]). It loads a compiled `.tmbin` model file and calls
`predict/2` on it — nothing else; there is no training code here (see
`docs/superpowers/specs/2026-09-23-tsetlin-nif-design.md` for the model
format contract).

[tsetlin-jl]: https://github.com/BooBSD/Tsetlin.jl
[rustler]: https://github.com/rusterlium/rustler

## Why a NIF, not pure Elixir

Built to run inference on resource-constrained targets — specifically a
[Nerves](https://nerves-project.org/) firmware for the Raspberry Pi Zero
(1st generation, ARMv6, single core, 512MB RAM). `TsetlinRunner.Target`
maps a Nerves `MIX_TARGET` to the matching Rust cross-compilation triple
so `mix compile`/`mix firmware` cross-compile the crate automatically.

## Usage

```elixir
{:ok, model} = TsetlinRunner.load("priv/models/my_model.tmbin")

packed = TsetlinRunner.pack_bits([true, false, true, ...])
{:ok, class} = TsetlinRunner.predict(model, packed)
```

`pack_bits/1` packs a list of booleans into the little-endian, LSB-first,
64-bit-chunk layout Tsetlin Machine models expect (matching `Tsetlin.jl`'s
own `TMInput` bit layout).

## Installation

Not published to Hex — reference it as a path or git dependency:

```elixir
def deps do
  [
    {:tsetlin_runner, path: "../tsetlin_runner"}
  ]
end
```
