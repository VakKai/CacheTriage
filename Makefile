CXXFLAGS ?= -O2
COMMON_FLAGS = -std=c++17 -Wall -Wextra -Wpedantic -Iinclude

.PHONY: all demo counterfactual pairwise test test-demo test-counterfactual test-pairwise clean
all: build/cachetriage

build/cachetriage: src/replay.cpp src/replay_support.hpp src/reuse_model.hpp src/trace_reader.hpp src/counting_allocator.hpp include/cachetriage/cache.hpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(LDFLAGS) -o $@ $<

demo: build/cachetriage_demo

build/cachetriage_demo: src/demo.cpp src/cache_budget.hpp src/counting_allocator.hpp src/pairwise_model.hpp src/eviction_history.hpp src/reuse_model.hpp src/trace_reader.hpp include/cachetriage/cache.hpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(LDFLAGS) -o $@ $< -lsqlite3

counterfactual: build/cachetriage_counterfactual

build/cachetriage_counterfactual: src/counterfactual.cpp src/trace_reader.hpp src/eviction_history.hpp include/cachetriage/cache.hpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(LDFLAGS) -o $@ $<

pairwise: build/cachetriage_pairwise

build/cachetriage_pairwise: src/pairwise_replay.cpp src/cache_budget.hpp src/pairwise_model.hpp src/eviction_history.hpp src/replay_support.hpp src/reuse_model.hpp src/counting_allocator.hpp src/trace_reader.hpp include/cachetriage/cache.hpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(LDFLAGS) -o $@ $<

build/check_compact_victims: tests/check_compact_victims.cpp src/pairwise_model.hpp src/eviction_history.hpp src/reuse_model.hpp src/trace_reader.hpp include/cachetriage/cache.hpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_FLAGS) -Isrc $(LDFLAGS) -o $@ $<

build/cache_test: tests/cache_test.cpp include/cachetriage/cache.hpp Makefile
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_FLAGS) $(LDFLAGS) -o $@ $<

test: build/cachetriage build/cache_test
	./build/cache_test
	./build/cachetriage 3 traces/example.txt > build/example.json
	diff -u tests/example.expected.json build/example.json
	./build/cachetriage 3 traces/example.txt --sieve-decisions > build/example.decisions.csv
	diff -u tests/example.decisions.csv build/example.decisions.csv
	./build/cachetriage 3 tests/oracle.txt --sieve-oracle 2 > build/oracle.json
	diff -u tests/oracle.expected.json build/oracle.json
	./build/cachetriage 3 tests/always_score.txt --always-score benchmarks/twitter-cluster052-reuse-v1.txt > build/always_score.json
	diff -u tests/always_score.expected.json build/always_score.json
	./build/cachetriage 3 tests/always_score.txt --triage benchmarks/twitter-cluster052-reuse-v1.txt 1 > build/gate.json
	diff -u tests/gate.expected.json build/gate.json

test-demo: build/cachetriage build/cachetriage_demo
	python3 -B tests/check_demo.py

test-counterfactual: build/cachetriage_counterfactual
	python3 -B tests/check_counterfactual.py

test-pairwise: build/cachetriage_pairwise build/cachetriage_demo build/check_compact_victims
	python3 -B tests/check_pairwise.py

clean:
	rm -rf build
