# Blockchain-Enabled Federated Learning for Zero-Day Detection

A comprehensive blockchain-based federated learning system for zero-day attack detection using transductive few-shot learning with test-time training.

## 🏗️ Project Structure

```
blockchain_federated_learning_project/
├── src/                                    # Source code
│   ├── main.py                            # Main system entry point
│   ├── models/                            # ML models
│   │   └── transductive_fewshot_model.py  # Transductive few-shot learning model
│   ├── clients/                           # Federated learning clients
│   │   └── blockchain_federated_clients.py
│   ├── coordinators/                      # Federated learning coordination
│   │   └── blockchain_fedavg_coordinator.py
│   ├── blockchain/                        # Blockchain integration
│   │   ├── blockchain_ipfs_integration.py
│   │   ├── blockchain_incentive_contract.py
│   │   ├── metamask_auth_system.py
│   │   └── incentive_provenance_system.py
│   ├── preprocessing/                     # Data preprocessing
│   │   └── blockchain_federated_unsw_preprocessor.py
│   └── visualization/                     # Performance visualization
│       └── performance_visualization.py
├── contracts/                             # Smart contracts
│   ├── CompleteFederatedLearning.sol
│   ├── FederatedLearning.sol
│   └── FederatedLearningIncentive.sol
├── scripts/                               # Deployment scripts
│   ├── deploy_blockchain_incentive_system.py
│   ├── deploy_complete_contracts.py
│   ├── deploy_minimal_contracts.py
│   ├── deploy_real_contracts.py
│   ├── deploy_simple_contracts.py
│   └── deploy_with_unlock.py
├── config/                                # Configuration files
│   └── deployed_contracts.json
├── tests/                                 # Test files
│   └── test_visualization.py
├── results/                               # Experiment results
├── performance_plots/                     # Generated plots
├── docs/                                  # Documentation
├── UNSW_NB15_training-set.csv            # Training dataset
├── UNSW_NB15_testing-set.csv             # Testing dataset
├── requirements_blockchain_fl.txt         # Python dependencies
└── README.md                              # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **Node.js** (for Ganache blockchain)
3. **IPFS** (for decentralized storage)
4. **MetaMask** (for wallet authentication)

### Installation

1. **Clone and setup:**

```bash
cd blockchain_federated_learning_project
pip install -r requirements_blockchain_fl.txt
```

2. **Start blockchain services:**

```bash
# Terminal 1: Start Ganache (Ethereum blockchain)
ganache-cli --accounts 10 --defaultBalanceEther 100 --gasLimit 0xfffffffffff --gasPrice 0x01

# Terminal 2: Start IPFS
ipfs daemon
```

3. **Deploy smart contracts:**

```bash
python scripts/deploy_complete_contracts.py
```

4. **Run the system:**

```bash
python src/main.py
```

## 🔧 System Components

### Core Features

- **Transductive Few-Shot Learning**: Graph-based similarity learning with attention mechanisms
- **Test-Time Training (TTT)**: Adaptive learning during inference
- **Blockchain Integration**: Ethereum smart contracts for model provenance
- **IPFS Storage**: Decentralized model storage with CID hashes
- **MetaMask Authentication**: Decentralized identity management
- **Incentive Mechanisms**: Token-based rewards for contributions
- **Performance Visualization**: Advanced plots with annotations

### Key Technologies

- **PyTorch**: Deep learning framework
- **Web3.py**: Ethereum blockchain interaction
- **IPFS**: Decentralized storage
- **Flower/FedML**: Federated learning coordination
- **UNSW-NB15**: Network intrusion detection dataset

## 📊 Performance Evaluation

The system includes comprehensive performance evaluation with:

- **Confusion Matrix**: Binary and multi-class classification
- **Precision, Recall, F1-Score**: Standard metrics
- **FPR (False Positive Rate)**: Security-specific metrics
- **Zero-Day Detection Rate**: Novel attack detection capability
- **Blockchain Metrics**: Transaction costs, gas usage, IPFS storage

### Visualization Features

- **Training History**: Loss and accuracy curves with trend analysis
- **Federated Rounds**: Client participation and model updates
- **Performance Comparison**: Base vs TTT model with improvement annotations
- **Client Performance**: Individual client metrics and contributions
- **Blockchain Metrics**: Transaction history and storage statistics
- **Comprehensive Reports**: Multi-panel analysis with professional styling

## 🔐 Security Features

- **Decentralized Transparency**: All model updates recorded on blockchain
- **Robust Provenance**: Immutable audit trail of model evolution
- **Incentive Mechanisms**: Fair reward distribution based on contributions
- **MetaMask Integration**: Secure wallet-based authentication
- **IPFS Verification**: Content-addressed storage with integrity checks

## 📈 Advanced Features

### Transductive Learning

- Graph-based similarity computation
- Multi-head attention mechanisms
- Joint optimization of support and test sets
- Test-time adaptation for zero-day detection

### Blockchain Integration

- Smart contract-based model aggregation
- Automatic reward distribution
- Transparent contribution tracking
- Decentralized model storage

### Performance Annotations

- Percentage improvement indicators
- Trend analysis with arrows
- Color-coded performance changes
- IEEE standard formatting

## 🧪 Testing

Run the visualization tests:

```bash
python tests/test_visualization.py
```

## 📝 Configuration

Key configuration parameters in `src/main.py`:

- `num_clients`: Number of federated learning clients
- `num_rounds`: Federated learning rounds
- `local_epochs`: Local training epochs per client
- `meta_epochs`: Meta-learning epochs
- `zero_day_attack`: Attack type for zero-day simulation

## 🔄 Workflow

1. **Data Preprocessing**: UNSW-NB15 dataset with zero-day holdout
2. **Model Training**: Transductive few-shot learning with meta-training
3. **Federated Learning**: Distributed training across clients
4. **Blockchain Integration**: Model updates stored on IPFS and blockchain
5. **Incentive Distribution**: Automatic reward calculation and distribution
6. **Performance Evaluation**: Comprehensive metrics and visualization

## 📚 Documentation

- **System Overview**: `docs/SYSTEM_OVERVIEW.md`
- **Incentive System**: `docs/INCENTIVE_SYSTEM_DOCUMENTATION.md`
- **Enhanced System**: `docs/ENHANCED_SYSTEM_SUMMARY.md`

## 🤝 Contributing

This is a research project for blockchain-enabled federated learning in cybersecurity applications.

## 📄 License

Research project - see individual component licenses.

## 🆘 Troubleshooting

### Common Issues

1. **Ganache Connection**: Ensure Ganache is running on port 8545
2. **IPFS Connection**: Start IPFS daemon with `ipfs daemon`
3. **Contract Deployment**: Check `config/deployed_contracts.json` for addresses
4. **Dataset**: Ensure UNSW-NB15 files are in project root

### Support

Check the logs for detailed error messages and ensure all prerequisites are installed.
