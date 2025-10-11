#!/usr/bin/env python3
"""
Test script for the fully decentralized federated learning system with PBFT consensus
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integration.fully_decentralized_system import run_fully_decentralized_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fully_decentralized_test.log')
    ]
)

logger = logging.getLogger(__name__)

async def test_fully_decentralized_system():
    """Test the fully decentralized system with 3 nodes"""
    logger.info("🧪 Starting Fully Decentralized System Test")
    logger.info("=" * 60)
    
    # Configuration for 3 nodes
    node_configs = [
        {
            'node_id': 'node_1',
            'port': 8765,
            'other_nodes': [('localhost', 8766), ('localhost', 8767)]
        },
        {
            'node_id': 'node_2', 
            'port': 8766,
            'other_nodes': [('localhost', 8765), ('localhost', 8767)]
        },
        {
            'node_id': 'node_3',
            'port': 8767,
            'other_nodes': [('localhost', 8765), ('localhost', 8766)]
        }
    ]
    
    logger.info("📊 Test Configuration:")
    logger.info(f"  Number of nodes: {len(node_configs)}")
    logger.info(f"  Training rounds: 5")
    logger.info(f"  Consensus algorithm: PBFT (f=0)")
    
    for i, config in enumerate(node_configs, 1):
        logger.info(f"  Node {i}: {config['node_id']} on port {config['port']}")
    
    try:
        # Run the test
        start_time = time.time()
        
        results = await run_fully_decentralized_training(
            num_rounds=5,
            node_configs=node_configs
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info("🎉 Test Completed Successfully!")
        logger.info("=" * 60)
        
        overall_metrics = results['overall_metrics']
        logger.info(f"⏱️  Total Test Time: {total_time:.2f} seconds")
        logger.info(f"📊 Overall Results:")
        logger.info(f"  Total Rounds: {overall_metrics['total_rounds']}")
        logger.info(f"  Successful Rounds: {overall_metrics['successful_rounds']}")
        logger.info(f"  Success Rate: {overall_metrics.get('success_rate', 0):.2%}")
        logger.info(f"  Average Consensus Time: {overall_metrics['average_consensus_time']:.2f}s")
        
        # Check if test passed
        success_rate = overall_metrics.get('success_rate', 0)
        if success_rate >= 0.8:  # 80% success rate threshold
            logger.info("✅ TEST PASSED: Success rate >= 80%")
            test_passed = True
        else:
            logger.warning(f"⚠️  TEST WARNING: Success rate {success_rate:.2%} < 80%")
            test_passed = False
        
        # Log detailed results
        logger.info("📈 Round-by-Round Details:")
        for round_data in results['rounds']:
            status = "✅" if round_data['consensus_success_rate'] >= 0.8 else "⚠️"
            logger.info(f"  {status} Round {round_data['round_number'] + 1}: "
                       f"{round_data['successful_nodes']}/{round_data['total_nodes']} nodes, "
                       f"Consensus Time: {round_data['average_consensus_time']:.2f}s, "
                       f"Success Rate: {round_data['consensus_success_rate']:.2%}")
        
        # Log node-specific metrics
        logger.info("🔍 Node-Specific Metrics:")
        for node_id, metrics in results['node_metrics'].items():
            consensus_metrics = metrics['consensus_metrics']
            logger.info(f"  {node_id}:")
            logger.info(f"    Total Rounds: {consensus_metrics['total_rounds']}")
            logger.info(f"    Successful Consensus: {consensus_metrics['successful_consensus']}")
            logger.info(f"    Average Consensus Time: {consensus_metrics['average_consensus_time']:.2f}s")
            logger.info(f"    Leader Elections: {consensus_metrics['leader_elections']}")
        
        # Save detailed results
        import json
        results_file = 'fully_decentralized_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Detailed results saved to '{results_file}'")
        
        return test_passed, results
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def main():
    """Main test function"""
    logger.info("🚀 Fully Decentralized Federated Learning System Test")
    logger.info("=" * 60)
    
    try:
        # Run the async test
        test_passed, results = asyncio.run(test_fully_decentralized_system())
        
        if test_passed:
            logger.info("🎉 ALL TESTS PASSED!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Some tests had warnings or failures")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

