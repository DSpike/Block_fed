#!/usr/bin/env python3
"""
Deploy Incentive Contract and Link to ERC20 Token
"""

import json
import time
from web3 import Web3
from solcx import compile_source, set_solc_version

# Set Solidity compiler version
set_solc_version('0.8.19')

# Ganache connection
GANACHE_URL = "http://localhost:8545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

def main():
    print("🚀 Deploying Incentive Contract...")
    
    # Check connection
    if not w3.is_connected():
        print("❌ Failed to connect to Ganache. Make sure it's running on port 8545")
        return
    
    print(f"✅ Connected to Ganache at {GANACHE_URL}")
    
    try:
        # Load existing ERC20 deployment
        with open('erc20_deployment.json', 'r') as f:
            erc20_deployment = json.load(f)
        
        token_address = erc20_deployment['contracts']['FederatedLearningToken']['address']
        print(f"📄 Using ERC20 token at: {token_address}")
        
        # Read simplified incentive contract
        incentive_source = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// ERC20 Token contract interface
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function distributeRewards(address[] calldata recipients, uint256[] calldata amounts) external;
}

contract FederatedLearningIncentive {
    address public owner;
    address public aggregator;
    address public tokenContract;
    
    uint256 public currentRound;
    uint256 public totalParticipants;
    
    mapping(address => bool) public participants;
    mapping(uint256 => mapping(address => bool)) public contributions;
    mapping(uint256 => mapping(address => uint256)) public rewards;
    
    event RoundStarted(uint256 indexed roundNumber, uint256 timestamp);
    event ContributionSubmitted(address indexed contributor, uint256 indexed roundNumber, uint256 timestamp);
    event RewardsDistributed(uint256 indexed roundNumber, address[] recipients, uint256[] amounts, uint256 timestamp);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator can call this function");
        _;
    }
    
    modifier onlyParticipant() {
        require(participants[msg.sender], "Only registered participants can call this function");
        _;
    }
    
    constructor(address _aggregator, address _tokenContract) {
        owner = msg.sender;
        aggregator = _aggregator;
        tokenContract = _tokenContract;
        currentRound = 0;
        totalParticipants = 0;
    }
    
    function registerParticipant(address participant) external onlyAggregator {
        require(!participants[participant], "Participant already registered");
        participants[participant] = true;
        totalParticipants++;
    }
    
    function submitContribution(uint256 roundNumber, uint256 accuracy, uint256 dataQuality) external onlyParticipant {
        require(roundNumber <= currentRound, "Invalid round number");
        require(!contributions[roundNumber][msg.sender], "Contribution already submitted");
        
        contributions[roundNumber][msg.sender] = true;
        
        // Calculate reward based on performance
        uint256 baseReward = 1000 * 10**18; // 1000 tokens
        uint256 performanceBonus = (accuracy * dataQuality * 10**18) / 10000; // Up to 1000 tokens bonus
        uint256 totalReward = baseReward + performanceBonus;
        
        rewards[roundNumber][msg.sender] = totalReward;
        
        emit ContributionSubmitted(msg.sender, roundNumber, block.timestamp);
    }
    
    function distributeRewards(uint256 roundNumber, address[] calldata recipients, uint256[] calldata amounts) external onlyAggregator {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length > 0, "No recipients");
        
        // Transfer tokens using ERC20 token contract
        IERC20(tokenContract).distributeRewards(recipients, amounts);
        
        emit RewardsDistributed(roundNumber, recipients, amounts, block.timestamp);
    }
    
    function startNewRound() external onlyAggregator {
        currentRound++;
        emit RoundStarted(currentRound, block.timestamp);
    }
    
    function getParticipantReward(uint256 roundNumber, address participant) external view returns (uint256) {
        return rewards[roundNumber][participant];
    }
    
    function isParticipant(address participant) external view returns (bool) {
        return participants[participant];
    }
    
    function getCurrentRound() external view returns (uint256) {
        return currentRound;
    }
}
'''
        
        # Compile incentive contract
        print("📝 Compiling Incentive contract...")
        compiled_sol = compile_source(
            incentive_source, 
            output_values=['abi', 'bin'],
            solc_version='0.8.19',
            allow_paths='.',
            import_remappings=[]
        )
        
        incentive_interface = compiled_sol['<stdin>:FederatedLearningIncentive']
        
        # Deploy incentive contract
        print("🚀 Deploying Incentive contract...")
        account = w3.eth.accounts[0]
        aggregator_address = w3.eth.accounts[0]
        
        # Build contract
        contract = w3.eth.contract(
            abi=incentive_interface['abi'],
            bytecode=incentive_interface['bin']
        )
        
        # Build constructor transaction
        constructor_tx = contract.constructor(aggregator_address, token_address).build_transaction({
            'from': account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
        
        # Deploy contract
        tx_hash = w3.eth.send_transaction(constructor_tx)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        incentive_address = tx_receipt.contractAddress
        
        print(f"✅ Incentive contract deployed at: {incentive_address}")
        
        # Set incentive contract in token contract
        print("🔗 Setting incentive contract in token contract...")
        token_abi = erc20_deployment['contracts']['FederatedLearningToken']['abi']
        token_contract = w3.eth.contract(address=token_address, abi=token_abi)
        
        tx_hash = token_contract.functions.setIncentiveContract(incentive_address).transact({
            'from': account,
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print("✅ Incentive contract set in token contract")
        
        # Update deployment info
        erc20_deployment['contracts']['FederatedLearningIncentive'] = {
            'address': incentive_address,
            'abi': incentive_interface['abi']
        }
        
        with open('erc20_deployment.json', 'w') as f:
            json.dump(erc20_deployment, f, indent=2)
        
        print(f"\n📄 Updated deployment info in: erc20_deployment.json")
        
        # Display summary
        print("\n🎉 INCENTIVE CONTRACT DEPLOYMENT COMPLETE!")
        print("=" * 60)
        print(f"ERC20 Token Address: {token_address}")
        print(f"Incentive Contract Address: {incentive_address}")
        print(f"Aggregator Address: {aggregator_address}")
        print(f"Deployer Account: {account}")
        print("=" * 60)
        
        # Test the system
        print("\n🧪 Testing the integrated system...")
        
        # Create contract instances
        incentive_contract = w3.eth.contract(address=incentive_address, abi=incentive_interface['abi'])
        
        # Register a participant
        participant = w3.eth.accounts[1]
        tx_hash = incentive_contract.functions.registerParticipant(participant).transact({
            'from': account,
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Registered participant: {participant}")
        
        # Start a new round
        tx_hash = incentive_contract.functions.startNewRound().transact({
            'from': account,
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print("✅ Started new round")
        
        # Submit contribution
        tx_hash = incentive_contract.functions.submitContribution(1, 85, 90).transact({
            'from': participant,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(participant)
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print("✅ Submitted contribution")
        
        # Check reward
        reward = incentive_contract.functions.getParticipantReward(1, participant).call()
        print(f"💰 Calculated reward: {reward / 10**18:.2f} FLT")
        
        # Distribute rewards
        recipients = [participant]
        amounts = [reward]
        tx_hash = incentive_contract.functions.distributeRewards(1, recipients, amounts).transact({
            'from': account,
            'gas': 300000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print("✅ Distributed rewards")
        
        # Check final balance
        final_balance = token_contract.functions.balanceOf(participant).call()
        print(f"💰 Final participant balance: {final_balance / 10**18:.2f} FLT")
        
        print("\n🎉 FULL SYSTEM TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

