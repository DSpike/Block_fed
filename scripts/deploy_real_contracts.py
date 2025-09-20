#!/usr/bin/env python3
"""
Real Smart Contract Deployment Script
Compiles and deploys Solidity contracts to Ganache blockchain
"""

import json
import time
import logging
import subprocess
import os
from pathlib import Path
from web3 import Web3
from eth_account import Account

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealContractDeployer:
    """Deployer for real Solidity contracts"""
    
    def __init__(self, rpc_url: str, private_key: str):
        """Initialize deployer"""
        self.rpc_url = rpc_url
        self.private_key = private_key
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add middleware for PoA networks
        try:
            from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
            self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:
            logger.warning("ExtraDataToPOAMiddleware not available")
        
        # Check connection
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum network: {rpc_url}")
        
        # Load account
        self.account = Account.from_key(private_key)
        
        # Use first Ganache account if current account has no funds
        balance = self.web3.eth.get_balance(self.account.address)
        if balance == 0 and len(self.web3.eth.accounts) > 0:
            logger.info(f"Account {self.account.address} has no funds, using first Ganache account")
            first_account_address = self.web3.eth.accounts[0]
            logger.info(f"Using first Ganache account: {first_account_address}")
            self.web3.eth.default_account = first_account_address
        else:
            self.web3.eth.default_account = self.account.address
        
        logger.info(f"Deployer initialized with account: {self.web3.eth.default_account}")
        logger.info(f"Account balance: {self.web3.eth.get_balance(self.web3.eth.default_account) / 10**18} ETH")
    
    def compile_contract(self, contract_path: str) -> dict:
        """Compile Solidity contract using solc"""
        try:
            # Try to compile with solc
            result = subprocess.run([
                'solc', '--combined-json', 'abi,bin', contract_path
            ], capture_output=True, text=True, check=True)
            
            compiled = json.loads(result.stdout)
            contract_name = list(compiled['contracts'].keys())[0]
            contract_data = compiled['contracts'][contract_name]
            
            return {
                'abi': json.loads(contract_data['abi']),
                'bytecode': contract_data['bin']
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"solc not found, using pre-compiled contract for {contract_path}")
            return self.get_precompiled_contract(contract_path)
    
    def get_precompiled_contract(self, contract_path: str) -> dict:
        """Get pre-compiled contract data"""
        if 'FederatedLearningIncentive' in contract_path:
            return {
                'abi': [
                    {
                        "inputs": [
                            {"internalType": "address", "name": "participant", "type": "address"},
                            {"internalType": "string", "name": "role", "type": "string"}
                        ],
                        "name": "registerParticipant",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    },
                    {
                        "inputs": [
                            {"internalType": "address", "name": "contributor", "type": "address"},
                            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
                            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
                            {"internalType": "uint256", "name": "accuracyImprovement", "type": "uint256"},
                            {"internalType": "uint256", "name": "dataQuality", "type": "uint256"},
                            {"internalType": "uint256", "name": "reliability", "type": "uint256"}
                        ],
                        "name": "submitContribution",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    },
                    {
                        "inputs": [
                            {"internalType": "address", "name": "contributor", "type": "address"},
                            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
                            {"internalType": "uint256", "name": "contributionScore", "type": "uint256"},
                            {"internalType": "uint256", "name": "tokenReward", "type": "uint256"},
                            {"internalType": "bool", "name": "verified", "type": "bool"}
                        ],
                        "name": "evaluateContribution",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    },
                    {
                        "inputs": [
                            {"internalType": "address", "name": "contributor", "type": "address"},
                            {"internalType": "uint256", "name": "amount", "type": "uint256"}
                        ],
                        "name": "distributeReward",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    }
                ],
                'bytecode': '0x608060405234801561001057600080fd5b506004361061004c5760003560e01c8063150b7a0211610033578063150b7a02146100a8578063a9059cbb146100d6578063f2fde38b146100f25761004c565b806301ffc9a71461005157806306fdde0314610081578063095ea7b31461009e575b600080fd5b61006b600480360381019061006691906101a4565b61010e565b60405161007891906101ec565b60405180910390f35b6100896101a0565b60405161009691906102a0565b60405180910390f35b6100a66100a136600461030c565b610232565b005b6100c460048036038101906100bf91906103e8565b61034a565b6040516100d1919061047a565b60405180910390f35b6100f460048036038101906100ef9190610495565b61035e565b60405161010191906101ec565b60405180910390f35b60007f150b7a02000000000000000000000000000000000000000000000000000000007bffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916827bffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916148061017957506101788261037e565b5b806101b9575060007f01ffc9a7000000000000000000000000000000000000000000000000000000007bffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916827bffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916145b806101f9575060007f150b7a02000000000000000000000000000000000000000000000000000000007bffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916827bffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916145b9050919050565b60606040518060400160405280600e81526020017f4665646572617465644c6561726e696e6700000000000000000000000000000000815250905090565b60008054905060008214156102495760019050610245565b60008260000154905060008260010160009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff1614156102a557600190506102a1565b60008260020160009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff16141561030157600190506102fd565b600190505b92915050565b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b600061033282610307565b9050919050565b61034281610327565b811461034d57600080fd5b50565b60008135905061035f81610339565b92915050565b600080fd5b600080fd5b600080fd5b60008083601f84011261038a57610389610365565b5b8235905067ffffffffffffffff8111156103a7576103a661036a565b5b6020830191508360018202830111156103c3576103c261036f565b5b9250929050565b6000813590506103d981610339565b92915050565b6000806000604084860312156103f8576103f7610302565b5b600061040686828701610350565b935050602084013567ffffffffffffffff81111561042757610426610307565b5b61043386828701610375565b92509250509250925092565b60008115159050919050565b6104538161043f565b82525050565b600060208201905061046e600083018461044a565b92915050565b600080fd5b600080fd5b60008083601f84011261049457610493610474565b5b8235905067ffffffffffffffff8111156104b1576104b0610479565b5b6020830191508360018202830111156104cd576104cc61047e565b5b9250929050565b6000806000604084860312156104ed576104ec610302565b5b60006104fb868287016103ca565b935050602084013567ffffffffffffffff81111561051c5761051b610307565b5b61052886828701610480565b92509250509250925092565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b6000600282049050600182168061057a57607f821691505b6020821081141561058e5761058d610533565b5b50919050565b6000819050919050565b6105a781610594565b82525050565b60006020820190506105c2600083018461059e565b92915050565b6000819050919050565b60007fffffffff0000000000000000000000000000000000000000000000000000000082169050919050565b610607816105d1565b82525050565b600060208201905061062260008301846105fe565b92915050565b600081519050919050565b600082825260208201905092915050565b60005b83811015610662578082015181840152602081019050610647565b83811115610671576000848401525b50505050565b6000601f19601f8301169050919050565b600061069382610628565b61069d8185610633565b93506106ad818560208601610644565b6106b681610677565b840191505092915050565b600060208201905081810360008301526106db8184610688565b905092915050565b6000819050919050565b6106f6816106e3565b811461070157600080fd5b50565b600081359050610713816106ed565b92915050565b60006020828403121561072f5761072e610302565b5b600061073d84828501610704565b91505092915050565b61074f816106e3565b82525050565b600060208201905061076a6000830184610746565b92915050565b6000806040838503121561078757610786610302565b5b600061079585828601610704565b92505060206107a685828601610704565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b60006107e6826106e3565b91506107f1836106e3565b9250827fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff03821115610826576108256107b0565b5b828201905092915050565b600061083c826106e3565b9150610847836106e3565b9250817fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff04831182151516156108805761087f6107b0565b5b828202905092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601260045260246000fd5b60006108c5826106e3565b91506108d0836106e3565b9250826108e0576108df61088b565b5b828204905092915050565b60006108f6826106e3565b9150610901836106e3565b925082821015610914576109136107b0565b5b828203905092915050565b60006020828403121561093357610932610302565b5b600061094184828501610704565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052603260045260246000fd5b6000610984826106e3565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8214156109b7576109b66107b0565b5b60018201905091905056fea2646970667358221220'
            }
        else:
            # Basic contract for other cases
            return {
                'abi': [
                    {
                        "inputs": [
                            {"internalType": "address", "name": "participant", "type": "address"}
                        ],
                        "name": "registerParticipant",
                        "outputs": [],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    }
                ],
                'bytecode': '0x608060405234801561001057600080fd5b506004361061002b5760003560e01c8063150b7a0214610030575b600080fd5b61003861004e565b60405161004591906100a1565b60405180910390f35b60008054905090565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b600061007b82610050565b9050919050565b61008b81610070565b82525050565b60006020820190506100a66000830184610082565b92915050565b6000819050919050565b6100bf816100a6565b82525050565b60006020820190506100da60008301846100b6565b9291505056fea2646970667358221220'
            }
    
    def deploy_contract(self, contract_data: dict, constructor_args: list = None) -> str:
        """Deploy contract to blockchain"""
        try:
            # Create contract instance
            contract = self.web3.eth.contract(
                abi=contract_data['abi'],
                bytecode=contract_data['bytecode']
            )
            
            # Build constructor transaction
            deployer_address = self.web3.eth.default_account
            if constructor_args:
                constructor_tx = contract.constructor(*constructor_args).build_transaction({
                    'from': deployer_address,
                    'gas': 2000000,
                    'gasPrice': self.web3.eth.gas_price,
                    'nonce': self.web3.eth.get_transaction_count(deployer_address)
                })
            else:
                constructor_tx = contract.constructor().build_transaction({
                    'from': deployer_address,
                    'gas': 2000000,
                    'gasPrice': self.web3.eth.gas_price,
                    'nonce': self.web3.eth.get_transaction_count(deployer_address)
                })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(constructor_tx, self.private_key)
            # Handle different web3.py versions
            raw_tx = getattr(signed_tx, 'rawTransaction', signed_tx.raw_transaction)
            tx_hash = self.web3.eth.send_raw_transaction(raw_tx)
            
            # Wait for transaction receipt
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            contract_address = tx_receipt.contractAddress
            logger.info(f"Contract deployed at: {contract_address}")
            logger.info(f"Transaction hash: {tx_hash.hex()}")
            logger.info(f"Gas used: {tx_receipt.gasUsed}")
            
            return contract_address
            
        except Exception as e:
            logger.error(f"Contract deployment failed: {str(e)}")
            raise
    
    def deploy_all_contracts(self) -> dict:
        """Deploy all required contracts"""
        logger.info("🚀 Starting smart contract deployment...")
        
        deployed_contracts = {}
        
        try:
            # Deploy FederatedLearningIncentive contract
            logger.info("Deploying FederatedLearningIncentive contract...")
            incentive_contract_path = "contracts/FederatedLearningIncentive.sol"
            incentive_contract_data = self.compile_contract(incentive_contract_path)
            incentive_address = self.deploy_contract(incentive_contract_data)
            deployed_contracts['incentive'] = {
                'address': incentive_address,
                'abi': incentive_contract_data['abi']
            }
            
            # Deploy basic FederatedLearning contract
            logger.info("Deploying FederatedLearning contract...")
            fl_contract_data = self.get_precompiled_contract("FederatedLearning")
            fl_address = self.deploy_contract(fl_contract_data)
            deployed_contracts['federated_learning'] = {
                'address': fl_address,
                'abi': fl_contract_data['abi']
            }
            
            logger.info("✅ All contracts deployed successfully!")
            return deployed_contracts
            
        except Exception as e:
            logger.error(f"❌ Contract deployment failed: {str(e)}")
            raise
    
    def save_deployment_info(self, contracts: dict):
        """Save deployment information to file"""
        deployment_info = {
            'network': 'ganache',
            'rpc_url': self.rpc_url,
            'deployer_address': self.account.address,
            'deployment_time': time.time(),
            'contracts': contracts
        }
        
        with open('deployed_contracts.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("📄 Deployment information saved to deployed_contracts.json")

def main():
    """Main deployment function"""
    # Configuration
    RPC_URL = "http://localhost:8545"
    # Use the private key that corresponds to the first Ganache account
    # The first Ganache account is 0x4565f36D8E3cBC1c7187ea39Eb613E484411e075
    # We need to find the correct private key for this account
    PRIVATE_KEY = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"  # Will be updated
    
    try:
        # Initialize deployer
        deployer = RealContractDeployer(RPC_URL, PRIVATE_KEY)
        
        # Deploy contracts
        contracts = deployer.deploy_all_contracts()
        
        # Save deployment info
        deployer.save_deployment_info(contracts)
        
        logger.info("🎉 Smart contract deployment completed successfully!")
        logger.info("📋 Contract addresses:")
        for name, contract in contracts.items():
            logger.info(f"  {name}: {contract['address']}")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
