// Copyright (c) The Diem Core Contributors
// SPDX-License-Identifier: Apache-2.0

//! This module implements an in-memory Sparse Merkle Tree that is similar to what we use in
//! storage to represent world state. This tree will store only a small portion of the state -- the
//! part of accounts that have been modified by uncommitted transactions. For example, if we
//! execute a transaction T_i on top of committed state and it modified account A, we will end up
//! having the following tree:
//! ```text
//!              S_i
//!             /   \
//!            o     y
//!           / \
//!          x   A
//! ```
//! where A has the new state of the account, and y and x are the siblings on the path from root to
//! A in the tree.
//!
//! This Sparse Merkle Tree is immutable once constructed. If the next transaction T_{i+1} modified
//! another account B that lives in the subtree at y, a new tree will be constructed and the
//! structure will look like the following:
//! ```text
//!                 S_i        S_{i+1}
//!                /   \      /       \
//!               /     y   /          \
//!              / _______/             \
//!             //                       \
//!            o                          y'
//!           / \                        / \
//!          x   A                      z   B
//! ```
//!
//! Using this structure, we are able to query the global state, taking into account the output of
//! uncommitted transactions. For example, if we want to execute another transaction T_{i+1}', we
//! can use the tree S_i. If we look for account A, we can find its new value in the tree.
//! Otherwise we know the account does not exist in the tree and we can fall back to storage. As
//! another example, if we want to execute transaction T_{i+2}, we can use the tree S_{i+1} that
//! has updated values for both account A and B.
//!
//! When we commit a transaction, for example T_i, we will first send its write set to storage.
//! Once the writes to storage complete, any node reachable from S_i will be available in storage.
//! Therefore we start from S_i and recursively drop its descendant. For internal or leaf nodes
//! (for example node o in the above example), we do not know if there are other nodes (for example
//! S_{i+1} in the above example) pointing to it, so we replace the node with a subtree node with
//! the same hash. This allows us to clean up memory as transactions are committed.
//!
//! This Sparse Merkle Tree serves a dual purpose. First, to support a leader based consensus
//! algorithm, we need to build a tree of transactions like the following:
//! ```text
//! Committed -> T5 -> T6  -> T7
//!              └---> T6' -> T7'
//!                    └----> T7"
//! ```
//! Once T5 is executed, we will have a tree that stores the modified portion of the state. Later
//! when we execute T6 on top of T5, the output of T5 can be visible to T6.
//!
//! Second, given this tree representation it is straightforward to compute the root hash of S_i
//! once T_i is executed. This allows us to verify the proofs we need when executing T_{i+1}.

// See https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=e9c4c53eb80b30d09112fcfb07d481e7
#![allow(clippy::let_and_return)]
// See https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=795cd4f459f1d4a0005a99650726834b
#![allow(clippy::while_let_loop)]

mod node;

#[cfg(test)]
mod sparse_merkle_test;

use self::node::{LeafNode, LeafValue, Node, SparseMerkleNode};
use diem_crypto::{
    hash::{CryptoHash, HashValueBitIterator, SPARSE_MERKLE_PLACEHOLDER_HASH},
    HashValue,
};
use diem_types::{
    account_state_blob::AccountStateBlob,
    proof::{SparseMerkleInternalNode, SparseMerkleLeafNode, SparseMerkleProof}
};
use itertools::sorted;
use std::sync::Arc;
use std::cmp;

/// `AccountStatus` describes the result of querying an account from this SparseMerkleTree.
#[derive(Debug, Eq, PartialEq)]
pub enum AccountStatus {
    /// The account exists in the tree, therefore we can give its value.
    ExistsInScratchPad(AccountStateBlob),

    /// The account does not exist in the tree, but exists in DB. This happens when the search
    /// reaches a leaf node that has the requested account, but the node has only the value hash
    /// because it was loaded into memory as part of a non-inclusion proof. When we go to DB we
    /// don't need to traverse the tree to find the same leaf, instead we can use the value hash to
    /// look up the account blob directly.
    ExistsInDB,

    /// The account does not exist in either the tree or DB. This happens when the search reaches
    /// an empty node, or a leaf node that has a different account.
    DoesNotExist,

    /// We do not know if this account exists or not and need to go to DB to find out. This happens
    /// when the search reaches a subtree node.
    Unknown,
}

/// The Sparse Merkle Tree implementation.
#[derive(Debug)]
pub struct SparseMerkleTree {
    root: Arc<SparseMerkleNode>,
}

impl SparseMerkleTree {
    /// Constructs a Sparse Merkle Tree with a root hash. This is often used when we restart and
    /// the scratch pad and the storage have identical state, so we use a single root hash to
    /// represent the entire state.
    pub fn new(root_hash: HashValue) -> Self {
        SparseMerkleTree {
            root: Arc::new(if root_hash != *SPARSE_MERKLE_PLACEHOLDER_HASH {
                SparseMerkleNode::new_subtree(root_hash)
            } else {
                SparseMerkleNode::new_empty()
            }),
        }
    }

    /// Constructs a new Sparse Merkle Tree after applying bulk of updates from a sequence of
    /// n transactions. Each update is of form (addr_hash, txn_id, blob), where 0 <= txn_id < n.
    /// Since the tree is immutable, the existing tree will remain the same and may share
    /// parts with the new, returned tree. Function also returns a vector of n-1 hashes, which
    /// would be the intedmediate root hashes if we serially applied the updates of transactions.
    pub fn bulk_update(
        &self,
        updates: Vec<(HashValue, usize, &AccountStateBlob)>,
        proof_reader: &impl ProofRead,
    ) -> Result<(Self, Vec<HashValue>), UpdateError> {
        let (root, hashes) = Self::bulk_update_existing_tree(Arc::clone(&self.root),
                                                             0,
                                                             &updates,
                                                             (0..updates.len()).collect(),
                                                             proof_reader)?;

        
        
        Ok((SparseMerkleTree { root },
            hashes.into_iter().skip(1).map(|(_, hash)| hash).collect()))
    }

    // Partition indices vector into two vectors, based on the bit value at bit_index of
    // the Hashvalue stored at updates[index].
    fn partition_indices(
        updates: &Vec<(HashValue, usize, &AccountStateBlob)>,
        indices: Vec<usize>,
        bit_index: usize
    ) -> (Vec<usize>, Vec<usize>) {
        indices
            .into_iter()
            .partition(|&index| updates[index].0.get_bit(bit_index))
    }

    // If updates vector, at specified, non-empty indices contains the same HashValues,
    // it's returned, otherwise None.
    fn all_keys_equal_to(
        updates: &Vec<(HashValue, usize, &AccountStateBlob)>,
        indices: &Vec<usize>,
    ) -> Option<HashValue> {
        match indices.len() {
            0 => None,
            _ => {
                let first_hashvalue = updates[indices[0]].0;
                if indices.into_iter().all(|&i| updates[i].0 == first_hashvalue) {
                    Some(first_hashvalue)
                } else {
                    None
                }
            }
        }
    }

    fn process_leaf_txn_hashes(
        before_hash: HashValue,
        key: HashValue,
        updates: &Vec<(HashValue, usize, &AccountStateBlob)>,
        indices: &Vec<usize>,
    ) -> (Arc<SparseMerkleNode>, Vec<(usize, HashValue)>) {
        let mut hashes = vec![(0, before_hash)];

        let txn_and_update_index: Vec<(usize, usize)> = sorted(
            indices
                .iter()
                .map(|&i| (updates[i].1 + 1, i)))
            .collect();

        txn_and_update_index
            .iter()
            .take(txn_and_update_index.len() - 1)
            .for_each(|&(num_txn, i)| {
                let value_hash = updates[i].2.hash();
                let leaf_hash = SparseMerkleLeafNode::new(key, value_hash).hash();
                hashes.push((num_txn, leaf_hash));
            });

        let (last_txn_num, last_update_index) = *txn_and_update_index.last().unwrap();
        
        let final_leaf = Arc::new(
            SparseMerkleNode::new_leaf(key,
                                       LeafValue::Blob(updates[last_update_index].2.clone()))
        );
        
        hashes.push((last_txn_num, final_leaf.read_lock().hash()));

        (final_leaf, hashes)
    }
          
    fn merge_txn_hashes(
        before_hash: HashValue,
        left_txn_hashes: Vec<(usize, HashValue)>,
        right_txn_hashes: Vec<(usize, HashValue)>,
    ) -> Vec<(usize, HashValue)> {
        let mut ret = vec![(0, before_hash)];
        
        let mut li = 1;
        let mut ri = 1;

        let next_txn_num = |i: usize, txn_hashes: &Vec<(usize, HashValue)>| {
            if i < txn_hashes.len() {
                txn_hashes[i].0
            } else {
                usize::MAX
            }
        };
                                                    
        let mut to_hash = vec![];
        while li < left_txn_hashes.len() || ri < right_txn_hashes.len() {
            let left_txn_num = next_txn_num(li, &left_txn_hashes);
            let right_txn_num = next_txn_num(ri, &right_txn_hashes);
            if left_txn_num <= right_txn_num {
                li = li + 1;
            }
            if right_txn_num <= left_txn_num {
                ri = ri + 1;
            }
            to_hash.push((cmp::min(left_txn_num, right_txn_num),
                          left_txn_hashes[li-1].1,
                          right_txn_hashes[ri-1].1));
        }

        ret.extend(
            to_hash
                // TODO: par_iter when appropriate.
                .iter()
                .map(|&(txn_num, left_hash, right_hash)|
                     (txn_num, SparseMerkleInternalNode::new(left_hash, right_hash).hash()))
        );
        ret
    }

    fn construct_internal_node(
        left_child: Arc<SparseMerkleNode>,
        right_child: Arc<SparseMerkleNode>,
        node_depth: usize,
        updates: &Vec<(HashValue, usize, &AccountStateBlob)>,
        subtree_update_indices: Vec<usize>,
        proof_reader: &impl ProofRead,
    ) -> Result<(Arc<SparseMerkleNode>, Vec<(usize, HashValue)>), UpdateError> {
        let (left_indices, right_indices) =
            Self::partition_indices(updates,
                                    subtree_update_indices,
                                    node_depth);
        
        let (left_child, left_txn_hashes) =
            Self::bulk_update_existing_tree(left_child,
                                            node_depth + 1,
                                            updates,
                                            left_indices,
                                            proof_reader)?;
        let (right_child, right_txn_hashes) =
            Self::bulk_update_existing_tree(right_child,
                                            node_depth + 1,
                                            updates,
                                            right_indices,
                                            proof_reader)?;

        let new_subtree_root =
            Arc::new(SparseMerkleNode::new_internal(left_child, right_child));
        let merged_txn_hashes = Self::merge_txn_hashes(new_subtree_root.read_lock().hash(),
                                                       left_txn_hashes,
                                                       right_txn_hashes);
        Ok((new_subtree_root, merged_txn_hashes))
    }
    
    // Returns the new subtree and a vector of pairs (txn_num, hash_value), where hash_value
    // would be the hash of the root of the subtree if only first txn_num many transactions
    // were applied. Only relevant txn_num values are included (i.e. if a transaction didn't
    // affect the given subtree, it will be skipped).
    fn bulk_update_existing_tree(
        subtree_root: Arc<SparseMerkleNode>,
        subtree_depth: usize,
        updates: &Vec<(HashValue, usize, &AccountStateBlob)>,
        subtree_update_indices: Vec<usize>,
        proof_reader: &impl ProofRead,
    ) -> Result<(Arc<SparseMerkleNode>, Vec<(usize, HashValue)>), UpdateError> {
        if subtree_update_indices.len() == 0 {
            let subtree_hash = subtree_root.read_lock().hash();
            return Ok((subtree_root, vec![(0, subtree_hash)]));
        }

        // Deal with subtree case.
        
        match &*subtree_root.read_lock() {
            Node::Internal(node) => Self::construct_internal_node(
                node.clone_left_child(),
                node.clone_right_child(),
                subtree_depth,
                updates,
                subtree_update_indices,
                proof_reader),
            Node::Empty => {
                match Self::all_keys_equal_to(updates, &subtree_update_indices) {
                    Some(key) => {
                        Ok(Self::process_leaf_txn_hashes(*SPARSE_MERKLE_PLACEHOLDER_HASH,
                                                         key,
                                                         updates,
                                                         &subtree_update_indices))
                    }
                    None => Self::construct_internal_node(
                        Arc::new(SparseMerkleNode::new_empty()),
                        Arc::new(SparseMerkleNode::new_empty()),
                        subtree_depth,
                        updates,
                        subtree_update_indices,
                        proof_reader),
                }
            }
            _ => {
                unreachable!("Not yet implemented.")
            }
        }
    }
    
    /// Constructs a new Sparse Merkle Tree as if we are updating the existing tree. Since the tree
    /// is immutable, the existing tree will remain the same and may share part of the tree with
    /// the new one.
    pub fn update(
        &self,
        updates: Vec<(HashValue, AccountStateBlob)>,
        proof_reader: &impl ProofRead,
    ) -> Result<Self, UpdateError> {
        let mut root = Arc::clone(&self.root);
        for (key, new_blob) in updates {
            root = Self::update_one(root, key, new_blob, proof_reader)?;
        }
        Ok(SparseMerkleTree { root })
    }

    fn update_one(
        root: Arc<SparseMerkleNode>,
        key: HashValue,
        new_blob: AccountStateBlob,
        proof_reader: &impl ProofRead,
    ) -> Result<Arc<SparseMerkleNode>, UpdateError> {
        let mut current_node = root;
        let mut bits = key.iter_bits();

        // Starting from root, traverse the tree according to key until we find a non-internal
        // node. Record all the bits and sibling nodes on the path.
        let mut bits_on_path = vec![];
        let mut siblings_on_path = vec![];
        loop {
            let next_node = if let Node::Internal(node) = &*current_node.read_lock() {
                let bit = bits.next().unwrap_or_else(|| {
                    // invariant of HashValueBitIterator
                    unreachable!("Tree is deeper than {} levels.", HashValue::LENGTH_IN_BITS)
                });
                bits_on_path.push(bit);
                if bit {
                    siblings_on_path.push(node.clone_left_child());
                    node.clone_right_child()
                } else {
                    siblings_on_path.push(node.clone_right_child());
                    node.clone_left_child()
                }
            } else {
                break;
            };
            current_node = next_node;
        }

        // Now we are at the bottom of the tree and current_node can be either a leaf, a subtree or
        // empty. We construct a new subtree like we are inserting the key here.
        let new_node =
            Self::construct_subtree_at_bottom(current_node, key, new_blob, bits, proof_reader)?;

        // Use the new node and all previous siblings on the path to construct the final tree.
        Ok(Self::construct_subtree(
            bits_on_path.into_iter().rev(),
            siblings_on_path.into_iter().rev(),
            new_node,
        ))
    }

    /// This function is called when we are trying to write (key, new_value) to the tree and have
    /// traversed the existing tree using some prefix of the key. We should have reached the bottom
    /// of the existing tree, so current_node cannot be an internal node. This function will
    /// construct a subtree using current_node, the new key-value pair and potentially the
    /// key-value pair in the proof.
    fn construct_subtree_at_bottom(
        current_node: Arc<SparseMerkleNode>,
        key: HashValue,
        new_blob: AccountStateBlob,
        remaining_bits: HashValueBitIterator,
        proof_reader: &impl ProofRead,
    ) -> Result<Arc<SparseMerkleNode>, UpdateError> {
        match &*current_node.read_lock() {
            Node::Internal(_) => {
                unreachable!("Reached an internal node at the bottom of the tree.")
            }
            Node::Leaf(node) => Ok(Self::construct_subtree_with_new_leaf(
                key,
                new_blob,
                node,
                HashValue::LENGTH_IN_BITS - remaining_bits.len(),
            )),
            Node::Subtree(_) => {
                // When the search reaches an Subtree node, we need proof to give us more
                // information about this part of the tree.
                let proof = proof_reader
                    .get_proof(key)
                    .ok_or(UpdateError::MissingProof)?;

                // Here the in-memory tree is identical to the tree in storage (we have only the
                // root hash of this subtree in memory). So we need to take into account the leaf
                // in the proof.
                let new_subtree = match proof.leaf() {
                    Some(existing_leaf) => Self::construct_subtree_with_new_leaf(
                        key,
                        new_blob,
                        &existing_leaf.into(),
                        proof.siblings().len(),
                    ),
                    None => Arc::new(SparseMerkleNode::new_leaf(key, LeafValue::Blob(new_blob))),
                };

                let num_remaining_bits = remaining_bits.len();
                let proof_length = proof.siblings().len();
                Ok(Self::construct_subtree(
                    remaining_bits
                        .rev()
                        .skip(HashValue::LENGTH_IN_BITS - proof_length),
                    proof
                        .siblings()
                        .iter()
                        .take(num_remaining_bits + proof_length - HashValue::LENGTH_IN_BITS)
                        .map(|sibling_hash| {
                            Arc::new(if *sibling_hash != *SPARSE_MERKLE_PLACEHOLDER_HASH {
                                SparseMerkleNode::new_subtree(*sibling_hash)
                            } else {
                                SparseMerkleNode::new_empty()
                            })
                        }),
                    new_subtree,
                ))
            }
            Node::Empty => {
                // When we reach an empty node, we just place the leaf node at the same position to
                // replace the empty node.
                Ok(Arc::new(SparseMerkleNode::new_leaf(
                    key,
                    LeafValue::Blob(new_blob),
                )))
            }
        }
    }

    /// Given key, new value, existing leaf and the distance from root to the existing leaf,
    /// constructs a new subtree that has either the new leaf or both nodes, depending on whether
    /// the key equals the existing leaf's key.
    ///
    /// 1. If the key equals the existing leaf's key, we simply need to update the leaf to the new
    ///    value and return it. For example, in the following case this function will return
    ///    `new_leaf`.
    ///  ``` text
    ///       o                    o
    ///      / \                  / \
    ///     o   o       =>       o   o
    ///    / \                  / \
    ///   o   existing_leaf    o   new_leaf
    ///  ```
    ///
    /// 2. Otherwise, we need to construct an "extension" for the common prefix, and at the end of
    ///    the extension a subtree for both keys. For example, in the following case we assume the
    ///    existing leaf's key starts with 010010 and key starts with 010011, and this function
    ///    will return `x`.
    /// ```text
    ///        o                              o             common_prefix_len = 5
    ///       / \                            / \            distance_from_root_to_existing_leaf = 2
    ///      o   o                          o   o           extension_len = common_prefix_len - distance_from_root_to_existing_leaf = 3
    ///     / \                            / \
    ///    o   existing_leaf    =>        o   x                 _
    ///                                      / \                ^
    ///                                     o   placeholder     |
    ///                                    / \                  |
    ///                                   o   placeholder   extension
    ///                                  / \                    |
    ///                       placeholder   o                   -
    ///                                    / \
    ///                       existing_leaf   new_leaf
    /// ```
    fn construct_subtree_with_new_leaf(
        key: HashValue,
        new_blob: AccountStateBlob,
        existing_leaf: &LeafNode,
        distance_from_root_to_existing_leaf: usize,
    ) -> Arc<SparseMerkleNode> {
        let new_leaf = Arc::new(SparseMerkleNode::new_leaf(key, LeafValue::Blob(new_blob)));

        if key == existing_leaf.key() {
            // This implies that `key` already existed and the proof is an inclusion proof.
            return new_leaf;
        }

        // This implies that `key` did not exist and was just created. The proof is a non-inclusion
        // proof. See above example for how extension_len is computed.
        let common_prefix_len = key.common_prefix_bits_len(existing_leaf.key());
        assert!(
            common_prefix_len >= distance_from_root_to_existing_leaf,
            "common_prefix_len: {}, distance_from_root_to_existing_leaf: {}",
            common_prefix_len,
            distance_from_root_to_existing_leaf,
        );
        let extension_len = common_prefix_len - distance_from_root_to_existing_leaf;
        Self::construct_subtree(
            key.iter_bits()
                .rev()
                .skip(HashValue::LENGTH_IN_BITS - common_prefix_len - 1)
                .take(extension_len + 1),
            std::iter::once(Arc::new(SparseMerkleNode::new_leaf(
                existing_leaf.key(),
                existing_leaf.value().clone(),
            )))
            .chain(std::iter::repeat(Arc::new(SparseMerkleNode::new_empty())).take(extension_len)),
            new_leaf,
        )
    }

    /// Constructs a subtree with a list of siblings and a leaf. For example, if `bits` are
    /// [false, false, true] and `siblings` are [a, b, c], the resulting subtree will look like:
    /// ```text
    ///          x
    ///         / \
    ///        c   o
    ///           / \
    ///          o   b
    ///         / \
    ///     leaf   a
    /// ```
    /// and this function will return `x`. Both `bits` and `siblings` start from the bottom.
    fn construct_subtree(
        bits: impl Iterator<Item = bool>,
        siblings: impl Iterator<Item = Arc<SparseMerkleNode>>,
        leaf: Arc<SparseMerkleNode>,
    ) -> Arc<SparseMerkleNode> {
        itertools::zip_eq(bits, siblings).fold(leaf, |previous_node, (bit, sibling)| {
            Arc::new(if bit {
                SparseMerkleNode::new_internal(sibling, previous_node)
            } else {
                SparseMerkleNode::new_internal(previous_node, sibling)
            })
        })
    }

    /// Queries a `key` in this `SparseMerkleTree`.
    pub fn get(&self, key: HashValue) -> AccountStatus {
        let mut current_node = Arc::clone(&self.root);
        let mut bits = key.iter_bits();

        loop {
            let next_node = if let Node::Internal(node) = &*current_node.read_lock() {
                match bits.next() {
                    Some(bit) => {
                        if bit {
                            node.clone_right_child()
                        } else {
                            node.clone_left_child()
                        }
                    }
                    None => panic!("Tree is deeper than {} levels.", HashValue::LENGTH_IN_BITS),
                }
            } else {
                break;
            };
            current_node = next_node;
        }

        let ret = match &*current_node.read_lock() {
            Node::Leaf(node) => {
                if key == node.key() {
                    match node.value() {
                        LeafValue::Blob(blob) => AccountStatus::ExistsInScratchPad(blob.clone()),
                        LeafValue::BlobHash(_) => AccountStatus::ExistsInDB,
                    }
                } else {
                    AccountStatus::DoesNotExist
                }
            }
            Node::Subtree(_) => AccountStatus::Unknown,
            Node::Empty => AccountStatus::DoesNotExist,
            Node::Internal(_) => {
                unreachable!("There is an internal node at the bottom of the tree.")
            }
        };
        ret
    }

    /// Returns the root hash of this tree.
    pub fn root_hash(&self) -> HashValue {
        self.root.read_lock().hash()
    }

    /// Prunes a tree by replacing every node reachable from root with a subtree node that has the
    /// same hash. If a node is empty or a subtree, we don't need to do anything. For example in
    /// the following case, if we drop `S_i`, we will replace o with a subtree node, then `o` no
    /// longer has pointers to its children `x` and `A`, so they will be dropped automatically.
    /// ```text
    ///            S_i        S_{i+1}                               S_{i+1}
    ///           /   \      /       \                             /       \
    ///          /     y   /          \          drop(S_i)        o         y'
    ///         / _______/             \         ========>                 / \
    ///        //                       \                                 z   B
    ///       o                          y'
    ///      / \                        / \
    ///     x   A                      z   B
    /// ```
    pub fn prune(&self) {
        let root = Arc::clone(&self.root);
        Self::prune_node(root);
    }

    fn prune_node(node: Arc<SparseMerkleNode>) {
        let mut writable_node = node.write_lock();
        let node_hash = writable_node.hash();

        match &*writable_node {
            Node::Empty => return,
            Node::Subtree(_) => return,
            Node::Internal(node) => {
                let left_child = node.clone_left_child();
                let right_child = node.clone_right_child();
                Self::prune_node(left_child);
                Self::prune_node(right_child);
            }
            Node::Leaf(_) => (),
        }

        *writable_node = Node::new_subtree(node_hash);
    }
}

impl Default for SparseMerkleTree {
    fn default() -> Self {
        SparseMerkleTree::new(*SPARSE_MERKLE_PLACEHOLDER_HASH)
    }
}

/// A type that implements `ProofRead` can provide proof for keys in persistent storage.
pub trait ProofRead {
    /// Gets verified proof for this key in persistent storage.
    fn get_proof(&self, key: HashValue) -> Option<&SparseMerkleProof>;
}

/// All errors `update` can possibly return.
#[derive(Debug, Eq, PartialEq)]
pub enum UpdateError {
    /// The update intends to insert a key that does not exist in the tree, so the operation needs
    /// proof to get more information about the tree, but no proof is provided.
    MissingProof,
}
