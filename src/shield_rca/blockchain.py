import hashlib
import json
import time
from loguru import logger

class SimpleBlockchain:
    def __init__(self):
        self.chain = []
        # Create genesis block
        self.create_block(proof=1, previous_hash='0', data="Genesis Block")

    def create_block(self, proof, previous_hash, data):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'proof': proof,
            'previous_hash': previous_hash,
            'data': data # This will be the hash of the event
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def store_event_hash(self, event_data):
        """
        Generates a SHA-256 hash of the event data and appends it to the chain.
        """
        # 1. Deterministic Hashing
        event_string = json.dumps(event_data, sort_keys=True)
        event_hash = hashlib.sha256(event_string.encode()).hexdigest()
        
        logger.info(f"[BLOCKCHAIN] Storing Event Hash: {event_hash}")
        
        # 2. Add to chain
        previous_block = self.get_previous_block()
        previous_hash = self.hash(previous_block)
        
        # Simple proof of work (dummy)
        proof = 1
        
        block = self.create_block(proof, previous_hash, event_hash)
        
        # Save to disk for persistence (hackathon friendly)
        self.save_chain()
        
        return event_hash, block['index']

    def save_chain(self):
        try:
            with open("blockchain.json", "w") as f:
                json.dump(self.chain, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save blockchain: {e}")

    def verify_event(self, event_data, stored_hash):
        """
        Recomputes hash and compares with blockchain.
        """
        event_string = json.dumps(event_data, sort_keys=True)
        recomputed_hash = hashlib.sha256(event_string.encode()).hexdigest()
        
        is_valid = recomputed_hash == stored_hash
        
        # Also check if it exists in the chain
        exists = False
        for block in self.chain:
            if block['data'] == stored_hash:
                exists = True
                break
                
        return is_valid and exists

# Global instance
blockchain_sys = SimpleBlockchain()
