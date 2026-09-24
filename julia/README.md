# julia/

The producer side of the `.tmbin` format this project's Rust NIF consumes:
`Export.jl`'s `export_tm(tm::TMClassifier, path)` takes an in-memory
`TMClassifier` (trained with the vendored [`BooBSD/Tsetlin.jl`][tsetlin-jl],
see `vendor/Tsetlin.jl/VENDORED.md` for exact commit/license) and writes it
to the binary layout documented in
`../docs/superpowers/specs/2026-09-23-tsetlin-nif-design.md` (section 3).

[tsetlin-jl]: https://github.com/BooBSD/Tsetlin.jl

## Usage

```julia
julia> include("Export.jl")
julia> tm = TMClassifier(x, Y, clauses_num, T, S, L, LF)  # from Tsetlin.jl
julia> train!(tm, data)                                    # ... however you train it
julia> export_tm(tm, "model.tmbin")
```

`export_tm` only needs the `TMClassifier` object itself — it doesn't read
or write `.tm` files.

## `.tm` files are not portable across projects

`Tsetlin.jl`'s own `save`/`load` (`.tm` files) use Julia's `Serialization`,
which ties a struct's type to the package that defined it when it was
serialized. A `.tm` file saved from inside some other project's package
(e.g. one that does `using MyProject` before training) can only be
`load`ed back from *that same package* — not from this directory's plain
`include("Export.jl")`, even though both vendor the identical `Tsetlin.jl`
source. If you need to export a model that already exists only as a `.tm`
file from another project, load it there and call `export_tm` from within
that project's own environment instead of copying the `.tm` file here.
