#!/usr/bin/env python3
"""
Real Gas Usage Data Collector for Blockchain Federated Learning
Captures actual gas consumption from blockchain transaction receipts
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RealGasTransaction:
    """Real gas transaction data from blockchain receipts"""
    transaction_hash: str
    transaction_type: str
    gas_used: int
    gas_limit: int
    gas_price: int
    block_number: int
    round_number: int
    client_id: Optional[str] = None
    ipfs_cid: Optional[str] = None
    timestamp: float = 0.0
    success: bool = True

class RealGasCollector:
    """
    Collects real gas usage data from blockchain transactions
    """
    
    def __init__(self):
        self.gas_transactions: List[RealGasTransaction] = []
        self.lock = threading.Lock()
        self.round_data: Dict[int, List[RealGasTransaction]] = defaultdict(list)
        
    def record_transaction(self, 
                          tx_hash: str,
                          tx_type: str,
                          gas_used: int,
                          gas_limit: int,
                          gas_price: int,
                          block_number: int,
                          round_number: int,
                          client_id: Optional[str] = None,
                          ipfs_cid: Optional[str] = None,
                          success: bool = True) -> None:
        """
        Record a real gas transaction
        
        Args:
            tx_hash: Transaction hash
            tx_type: Type of transaction (Client Update, Model Aggregation, etc.)
            gas_used: Actual gas consumed
            gas_limit: Gas limit set for transaction
            gas_price: Gas price in wei
            block_number: Block number where transaction was mined
            round_number: Federated learning round number
            client_id: Client identifier (if applicable)
            ipfs_cid: IPFS content identifier (if applicable)
            success: Whether transaction was successful
        """
        with self.lock:
            transaction = RealGasTransaction(
                transaction_hash=tx_hash,
                transaction_type=tx_type,
                gas_used=gas_used,
                gas_limit=gas_limit,
                gas_price=gas_price,
                block_number=block_number,
                round_number=round_number,
                client_id=client_id,
                ipfs_cid=ipfs_cid,
                timestamp=time.time(),
                success=success
            )
            
            self.gas_transactions.append(transaction)
            self.round_data[round_number].append(transaction)
            
            logger.info(f"Recorded real gas transaction: {tx_type} - Gas: {gas_used}, Block: {block_number}")
    
    def get_round_gas_data(self, round_number: int) -> Dict[str, Any]:
        """
        Get gas usage data for a specific round
        
        Args:
            round_number: Round number
            
        Returns:
            Dictionary with gas usage statistics for the round
        """
        with self.lock:
            round_transactions = self.round_data.get(round_number, [])
            
            if not round_transactions:
                return {
                    'round_number': round_number,
                    'total_transactions': 0,
                    'total_gas_used': 0,
                    'average_gas_used': 0,
                    'transaction_types': {},
                    'transactions': []
                }
            
            # Calculate statistics
            total_gas_used = sum(tx.gas_used for tx in round_transactions)
            average_gas_used = total_gas_used / len(round_transactions)
            
            # Group by transaction type
            transaction_types = defaultdict(list)
            for tx in round_transactions:
                transaction_types[tx.transaction_type].append(tx.gas_used)
            
            # Calculate averages by type
            type_averages = {}
            for tx_type, gas_values in transaction_types.items():
                type_averages[tx_type] = {
                    'count': len(gas_values),
                    'total_gas': sum(gas_values),
                    'average_gas': sum(gas_values) / len(gas_values),
                    'min_gas': min(gas_values),
                    'max_gas': max(gas_values)
                }
            
            return {
                'round_number': round_number,
                'total_transactions': len(round_transactions),
                'total_gas_used': total_gas_used,
                'average_gas_used': average_gas_used,
                'transaction_types': type_averages,
                'transactions': [asdict(tx) for tx in round_transactions]
            }
    
    def get_all_gas_data(self) -> Dict[str, Any]:
        """
        Get comprehensive gas usage data for all rounds
        
        Returns:
            Dictionary with complete gas usage analysis
        """
        with self.lock:
            if not self.gas_transactions:
                return {
                    'total_transactions': 0,
                    'total_gas_used': 0,
                    'rounds': {},
                    'transaction_types': {},
                    'summary': {}
                }
            
            # Calculate overall statistics
            total_gas_used = sum(tx.gas_used for tx in self.gas_transactions)
            total_transactions = len(self.gas_transactions)
            average_gas_used = total_gas_used / total_transactions
            
            # Group by transaction type
            transaction_types = defaultdict(list)
            for tx in self.gas_transactions:
                transaction_types[tx.transaction_type].append(tx.gas_used)
            
            # Calculate type statistics
            type_stats = {}
            for tx_type, gas_values in transaction_types.items():
                type_stats[tx_type] = {
                    'count': len(gas_values),
                    'total_gas': sum(gas_values),
                    'average_gas': sum(gas_values) / len(gas_values),
                    'min_gas': min(gas_values),
                    'max_gas': max(gas_values),
                    'percentage': (sum(gas_values) / total_gas_used) * 100
                }
            
            # Get round data (avoid deadlock by not calling get_round_gas_data while holding lock)
            rounds_data = {}
            for round_num in self.round_data.keys():
                round_transactions = self.round_data.get(round_num, [])
                if round_transactions:
                    total_gas = sum(tx.gas_used for tx in round_transactions)
                    rounds_data[round_num] = {
                        'round_number': round_num,
                        'total_transactions': len(round_transactions),
                        'total_gas_used': total_gas,
                        'average_gas_used': total_gas / len(round_transactions) if round_transactions else 0,
                        'transaction_types': {},
                        'transactions': [{'transaction_hash': tx.transaction_hash, 'transaction_type': tx.transaction_type, 'gas_used': tx.gas_used, 'block_number': tx.block_number, 'round_number': tx.round_number, 'client_id': tx.client_id, 'ipfs_cid': tx.ipfs_cid, 'timestamp': tx.timestamp, 'success': tx.success} for tx in round_transactions]
                    }
                else:
                    rounds_data[round_num] = {
                        'round_number': round_num,
                        'total_transactions': 0,
                        'total_gas_used': 0,
                        'average_gas_used': 0,
                        'transaction_types': {},
                        'transactions': []
                    }
            
            return {
                'total_transactions': total_transactions,
                'total_gas_used': total_gas_used,
                'average_gas_used': average_gas_used,
                'rounds': rounds_data,
                'transaction_types': type_stats,
                'summary': {
                    'most_expensive_operation': max(type_stats.items(), key=lambda x: x[1]['average_gas'])[0],
                    'most_frequent_operation': max(type_stats.items(), key=lambda x: x[1]['count'])[0],
                    'gas_efficiency': (total_gas_used / sum(tx.gas_limit for tx in self.gas_transactions)) * 100,
                    'total_rounds': len(self.round_data)
                }
            }
    
    def export_to_json(self, filename: str) -> None:
        """
        Export gas usage data to JSON file
        
        Args:
            filename: Output filename
        """
        with self.lock:
            data = self.get_all_gas_data()
            data['export_timestamp'] = time.time()
            data['export_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Real gas usage data exported to {filename}")
    
    def clear_data(self) -> None:
        """Clear all collected gas data"""
        with self.lock:
            self.gas_transactions.clear()
            self.round_data.clear()
            logger.info("Real gas usage data cleared")

# Global instance for system-wide gas collection
real_gas_collector = RealGasCollector()
