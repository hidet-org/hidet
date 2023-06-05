#pragma once
#include <list>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <hidet/runtime/common.h>

struct Region {
    int64_t start;
    int64_t size;
};

struct MemoryPlanner {
    std::list<Region> regions;
    std::unordered_map<int64_t, int64_t> size_map;
};

static std::vector<MemoryPlanner> memory_planners;

static void memory_planner_init(int idx) {
    if(memory_planners.size() <= idx) {
        memory_planners.resize(idx + 1);
    }
    memory_planners[idx].size_map.clear();
    memory_planners[idx].regions.clear();
    memory_planners[idx].regions.push_back({0, -1});
}

static int64_t memory_planner_allocate(int idx, int64_t size) {
    MemoryPlanner &memory_planner = memory_planners[idx];

    if(size == 0) {
        return -1;
    }

    size = (size + 127) / 128 * 128;    // ceil to 128 bytes
    for (auto it = memory_planner.regions.begin(); it != memory_planner.regions.end(); ++it) {
        if (it->size >= size) {
            auto region = *it;
            if (region.size > size) {
                memory_planner.regions.insert(it, {region.start + size, region.size - size});
            }
            memory_planner.regions.erase(it);
            auto ret = region.start;
            memory_planner.size_map[ret] = size;
            return ret;
        } else if (it->size == -1) {
            int64_t start = it->start;
            it->start += size;
            memory_planner.size_map[start] = size;
            return start;
        }
    }
    assert(false);
    return 0;
}

static void memory_planner_free(int idx, int64_t ptr) {
    MemoryPlanner &memory_planner = memory_planners[idx];

    if(ptr == -1) {
        return;
    }

    int64_t start = ptr;
    int64_t size = memory_planner.size_map[ptr];
    auto it = memory_planner.regions.begin();
    while(it != memory_planner.regions.end() && it->start <= start)
        it++;
    if(it == memory_planner.regions.begin()) {
        if(start + size == it->start) {
            it->start = start;
            if(it->size != -1) {
                it->size += size;
            }
        } else {
            memory_planner.regions.insert(it, {start, size});
        }
    } else {
        auto pit = it;
        pit--;
        if(start + size == it->start && start == pit->start + pit->size) {
            it->start = pit->start;
            if(it->size != -1) {
                it->size += pit->size + size;
            }
            memory_planner.regions.erase(pit);
        } else if (start + size == it->start){
            it->start = start;
            if (it->size != -1) {
                it->size += size;
            }
        } else if (start == pit->start + pit->size) {
            pit->size += size;
        } else {
            memory_planner.regions.insert(it, {start, size});
        }
    }
}

static int64_t memory_planner_used(int idx) {
    MemoryPlanner &memory_planner = memory_planners[idx];
    auto riter = memory_planner.regions.rbegin();
    return riter->start;
}
