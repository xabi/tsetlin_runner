# Exports a compiled TMClassifier (trained with the vendored BooBSD/Tsetlin.jl,
# see vendor/Tsetlin.jl/VENDORED.md) to the portable `.tmbin` binary format
# this project's Rust NIF parses (format contract:
# docs/superpowers/specs/2026-09-23-tsetlin-nif-design.md, section 3). Any
# change here must stay in lockstep with that spec.
#
# Usage (from this directory, or any script that `include`s this file):
#
#   include("Export.jl")
#   tm = Tsetlin.load("model.tm")   # a (TMClassifier, accuracy) tuple or bare TMClassifier
#   export_tm(tm isa Tuple ? tm[1] : tm, "model.tmbin")

include(joinpath(@__DIR__, "vendor", "Tsetlin.jl", "src", "Tsetlin.jl"))
using .Tsetlin

_write_block(io, clauses::Tsetlin.TMClauses) = begin
    write(io, clauses.positive_included_literals)
    write(io, clauses.positive_included_literals_inverted)
    write(io, clauses.negative_included_literals)
    write(io, clauses.negative_included_literals_inverted)
end

"""
    export_tm(tm::TMClassifier, path::AbstractString)

Compiles `tm` (dropping training-only automata state via `compile`) and
writes it to `path` in the little-endian `.tmbin` layout: magic "TSTM",
version, kind (0=Bool/1=general), clause_size, chunks_size, classes_num,
ta_clauses, LF, classes[], then one block per class (or one shared block
for the Bool-specialized classifier) of the four `*_included_literals`
matrices, in Julia's native column-major order (already the layout the
consumer's column-slicing expects, so no transposition is needed).
"""
function export_tm(tm::TMClassifier, path::AbstractString)
    ctm = Tsetlin.compile(tm)
    is_bool = ctm.clauses isa Tsetlin.TMClauses
    kind = is_bool ? 0x00 : 0x01
    blocks = is_bool ? (ctm.clauses,) : ctm.clauses
    first_block = first(blocks)
    chunks_size = size(first_block.positive_included_literals, 1)
    ta_clauses = size(first_block.positive_included_literals, 2)
    classes = is_bool ? Int64[1, 0] : Int64.(ctm.classes)

    open(path, "w") do io
        write(io, "TSTM")
        write(io, UInt32(1))               # version
        write(io, kind)
        write(io, UInt32(ctm.clause_size))
        write(io, UInt32(chunks_size))
        write(io, UInt32(length(classes)))
        write(io, UInt32(ta_clauses))
        write(io, Int64(ctm.LF))
        for c in classes
            write(io, Int64(c))
        end
        for block in blocks
            _write_block(io, block)
        end
    end
    return path
end
