
building_blocks = Vector{Expr}([quote x end, quote y end, quote z end, quote β end, quote σ end, quote ρ end]);
building_block_types = Vector{Type}([Number, Number, Number, Number, Number, Number, Number,]);
operations = Vector([+, -, *]);
operation_input_types = Vector([(Number, Number), (Number, Number), (Number, Number)])

function compute_MSE(model)