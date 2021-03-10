initSidebarItems({"static":[["ACCUMULATOR_PLACEHOLDER_HASH","Placeholder hash of `Accumulator`."],["GENESIS_BLOCK_ID","Genesis block id is used as a parent of the very first block executed by the executor."],["PRE_GENESIS_BLOCK_ID","Block id reserved as the id of parent block of the genesis block."],["SPARSE_MERKLE_PLACEHOLDER_HASH","Placeholder hash of `SparseMerkleTree`."]],"struct":[["EventAccumulatorHasher","The hasher used to compute the hash of an internal node in the event accumulator."],["HashValue","Output value of our hash function. Intentionally opaque for safety and modularity."],["HashValueBitIterator","An iterator over `HashValue` that generates one bit for each iteration."],["SparseMerkleInternalHasher","The hasher used to compute the hash of an internal node in the Sparse Merkle Tree."],["TestOnlyHasher","The hasher used only for testing. It doesn't have a salt."],["TransactionAccumulatorHasher","The hasher used to compute the hash of an internal node in the transaction accumulator."],["VoteProposalHasher","The hasher used to compute the hash of an internal node in the transaction accumulator."]],"trait":[["CryptoHash","A type that can be cryptographically hashed to produce a `HashValue`."],["CryptoHasher","A trait for representing the state of a cryptographic hasher."],["TestOnlyHash","Provides a test_only_hash() method that can be used in tests on types that implement `serde::Serialize`."]]});