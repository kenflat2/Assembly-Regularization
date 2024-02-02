using DataStructures
import DataStructures.enqueue!

struct FixedSizePriorityQueue{K, V}
    pq::PriorityQueue{K, V}
    max_size::Int
end

function FixedSizePriorityQueue{K, V}(max_size::Int) where K where V
    pq = PriorityQueue{K, V}(Base.Order.Reverse)
    return FixedSizePriorityQueue(pq, max_size)
end

function enqueue!(q::FixedSizePriorityQueue{K,V}, key::K, val::V) where K where V
    if length(q.pq) < q.max_size || val < minimum(values(q.pq))
        enqueue!(q.pq, key, val)
        if length(q.pq) > q.max_size
            dequeue!(q.pq)
        end
    end
end
