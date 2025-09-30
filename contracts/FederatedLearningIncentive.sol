// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FederatedLearningIncentive
 * @dev Smart contract for blockchain-based incentive mechanism in federated learning
 * Handles model contribution evaluation, token rewards, and transparent audit trails
 */
// ERC20 Token contract interface
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function distributeRewards(address[] calldata recipients, uint256[] calldata amounts) external;
}

contract FederatedLearningIncentive {
    
    // ERC20 Token contract address
    address public tokenContract;
    
    // Events
    event ModelContributionSubmitted(
        address indexed contributor,
        uint256 indexed roundNumber,
        bytes32 modelHash,
        uint256 accuracyImprovement,
        uint256 dataQuality,
        uint256 reliability,
        uint256 timestamp
    );
    
    event ContributionEvaluated(
        address indexed contributor,
        uint256 indexed roundNumber,
        uint256 contributionScore,
        uint256 tokenReward,
        bool verified,
        uint256 timestamp
    );
    
    event TokenRewardDistributed(
        address indexed recipient,
        uint256 amount,
        uint256 indexed roundNumber,
        bytes32 transactionHash,
        uint256 timestamp
    );
    
    event AggregationCompleted(
        uint256 indexed roundNumber,
        uint256 totalContributors,
        uint256 totalRewards,
        uint256 timestamp
    );
    
    event ReputationUpdated(
        address indexed participant,
        uint256 oldReputation,
        uint256 newReputation,
        uint256 indexed roundNumber,
        uint256 timestamp
    );
    
    // Structs
    struct ModelContribution {
        address contributor;
        uint256 roundNumber;
        bytes32 modelHash;
        uint256 accuracyImprovement;  // Percentage improvement (0-100)
        uint256 dataQuality;          // Data quality score (0-100)
        uint256 reliability;          // Model reliability score (0-100)
        uint256 timestamp;
        bool evaluated;
        uint256 contributionScore;
        uint256 tokenReward;
        bool verified;
    }
    
    struct Participant {
        address participantAddress;
        uint256 reputation;           // Reputation score (0-1000)
        uint256 totalContributions;
        uint256 totalRewards;
        uint256 lastActivity;
        bool isActive;
        uint256 verificationCount;
    }
    
    struct RoundData {
        uint256 roundNumber;
        uint256 totalContributors;
        uint256 totalRewards;
        uint256 averageAccuracy;
        uint256 timestamp;
        bool completed;
        mapping(address => bool) contributors;
    }
    
    // State variables
    mapping(address => Participant) public participants;
    mapping(uint256 => RoundData) public rounds;
    mapping(uint256 => mapping(address => ModelContribution)) public contributions;
    
    address public owner;
    address public aggregator;
    uint256 public currentRound;
    uint256 public totalParticipants;
    
    // Token and reward parameters
    uint256 public constant BASE_REWARD = 100; // Base reward in tokens
    uint256 public constant MAX_REWARD = 1000; // Maximum reward per contribution
    uint256 public constant MIN_REPUTATION = 100; // Minimum reputation to participate
    
    // Contribution evaluation weights
    uint256 public constant ACCURACY_WEIGHT = 40; // 40% weight for accuracy
    uint256 public constant QUALITY_WEIGHT = 30;  // 30% weight for data quality
    uint256 public constant RELIABILITY_WEIGHT = 30; // 30% weight for reliability
    
    // Reputation parameters
    uint256 public constant REPUTATION_DECAY = 95; // 5% decay per round
    uint256 public constant REPUTATION_BOOST = 10; // Reputation boost for good contributions
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator can call this function");
        _;
    }
    
    modifier onlyActiveParticipant() {
        require(participants[msg.sender].isActive, "Participant not active");
        require(participants[msg.sender].reputation >= MIN_REPUTATION, "Insufficient reputation");
        _;
    }
    
    modifier validRound(uint256 roundNumber) {
        require(roundNumber <= currentRound, "Invalid round number");
        _;
    }
    
    // Constructor
    constructor(address _aggregator, address _tokenContract) {
        owner = msg.sender;
        aggregator = _aggregator;
        tokenContract = _tokenContract;
        currentRound = 0;
        totalParticipants = 0;
    }
    
    /**
     * @dev Set the token contract address
     * @param _tokenContract Address of the ERC20 token contract
     */
    function setTokenContract(address _tokenContract) external onlyAggregator {
        tokenContract = _tokenContract;
    }
    
    /**
     * @dev Register a new participant
     * @param participantAddress Address of the participant
     */
    function registerParticipant(address participantAddress) external onlyOwner {
        require(!participants[participantAddress].isActive, "Participant already registered");
        
        participants[participantAddress] = Participant({
            participantAddress: participantAddress,
            reputation: 500, // Initial reputation (middle of 0-1000 scale)
            totalContributions: 0,
            totalRewards: 0,
            lastActivity: block.timestamp,
            isActive: true,
            verificationCount: 0
        });
        
        totalParticipants++;
    }
    
    /**
     * @dev Submit model contribution for evaluation
     * @param roundNumber Current round number
     * @param modelHash Hash of the model parameters
     * @param accuracyImprovement Accuracy improvement percentage (0-100)
     * @param dataQuality Data quality score (0-100)
     * @param reliability Model reliability score (0-100)
     */
    function submitModelContribution(
        uint256 roundNumber,
        bytes32 modelHash,
        uint256 accuracyImprovement,
        uint256 dataQuality,
        uint256 reliability
    ) external onlyActiveParticipant {
        require(roundNumber == currentRound, "Invalid round number");
        require(accuracyImprovement <= 100, "Invalid accuracy improvement");
        require(dataQuality <= 100, "Invalid data quality");
        require(reliability <= 100, "Invalid reliability");
        require(!contributions[roundNumber][msg.sender].evaluated, "Contribution already submitted");
        
        // Create contribution record
        contributions[roundNumber][msg.sender] = ModelContribution({
            contributor: msg.sender,
            roundNumber: roundNumber,
            modelHash: modelHash,
            accuracyImprovement: accuracyImprovement,
            dataQuality: dataQuality,
            reliability: reliability,
            timestamp: block.timestamp,
            evaluated: false,
            contributionScore: 0,
            tokenReward: 0,
            verified: false
        });
        
        // Update participant activity
        participants[msg.sender].lastActivity = block.timestamp;
        participants[msg.sender].totalContributions++;
        
        // Add to round contributors
        rounds[roundNumber].contributors[msg.sender] = true;
        
        emit ModelContributionSubmitted(
            msg.sender,
            roundNumber,
            modelHash,
            accuracyImprovement,
            dataQuality,
            reliability,
            block.timestamp
        );
    }
    
    /**
     * @dev Evaluate model contribution and calculate rewards
     * @param contributor Address of the contributor
     * @param roundNumber Round number
     * @param verificationResult Whether the contribution is verified
     */
    function evaluateContribution(
        address contributor,
        uint256 roundNumber,
        bool verificationResult
    ) external onlyAggregator validRound(roundNumber) {
        require(contributions[roundNumber][contributor].contributor != address(0), "Contribution not found");
        require(!contributions[roundNumber][contributor].evaluated, "Contribution already evaluated");
        
        ModelContribution storage contribution = contributions[roundNumber][contributor];
        
        // Calculate contribution score
        uint256 contributionScore = calculateContributionScore(
            contribution.accuracyImprovement,
            contribution.dataQuality,
            contribution.reliability
        );
        
        // Calculate token reward
        uint256 tokenReward = calculateTokenReward(contributor, contributionScore, verificationResult);
        
        // Update contribution record
        contribution.evaluated = true;
        contribution.contributionScore = contributionScore;
        contribution.tokenReward = tokenReward;
        contribution.verified = verificationResult;
        
        // Update participant rewards
        participants[contributor].totalRewards += tokenReward;
        
        // Update reputation
        updateReputation(contributor, contributionScore, verificationResult);
        
        emit ContributionEvaluated(
            contributor,
            roundNumber,
            contributionScore,
            tokenReward,
            verificationResult,
            block.timestamp
        );
    }
    
    /**
     * @dev Distribute token rewards to verified contributors
     * @param roundNumber Round number
     * @param recipients Array of recipient addresses
     * @param amounts Array of reward amounts
     */
    function distributeTokenRewards(
        uint256 roundNumber,
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external onlyAggregator validRound(roundNumber) {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length > 0, "No recipients");
        
        uint256 totalDistributed = 0;
        
        // Validate all contributions first
        for (uint256 i = 0; i < recipients.length; i++) {
            address recipient = recipients[i];
            uint256 amount = amounts[i];
            
            require(contributions[roundNumber][recipient].verified, "Contribution not verified");
            require(contributions[roundNumber][recipient].tokenReward == amount, "Amount mismatch");
            
            totalDistributed += amount;
        }
        
        // Transfer tokens using ERC20 token contract (single call for all recipients)
        IERC20(tokenContract).distributeRewards(recipients, amounts);
        
        // Emit events for each recipient
        for (uint256 i = 0; i < recipients.length; i++) {
            address recipient = recipients[i];
            uint256 amount = amounts[i];
            
            emit TokenRewardDistributed(
                recipient,
                amount,
                roundNumber,
                keccak256(abi.encodePacked(recipient, amount, roundNumber, block.timestamp)),
                block.timestamp
            );
        }
        
        // Update round data
        rounds[roundNumber].totalRewards = totalDistributed;
        
        emit AggregationCompleted(
            roundNumber,
            recipients.length,
            totalDistributed,
            block.timestamp
        );
    }
    
    /**
     * @dev Complete aggregation round
     * @param roundNumber Round number to complete
     * @param averageAccuracy Average accuracy of the round
     */
    function completeAggregationRound(
        uint256 roundNumber,
        uint256 averageAccuracy
    ) external onlyAggregator validRound(roundNumber) {
        require(!rounds[roundNumber].completed, "Round already completed");
        
        rounds[roundNumber].roundNumber = roundNumber;
        rounds[roundNumber].averageAccuracy = averageAccuracy;
        rounds[roundNumber].timestamp = block.timestamp;
        rounds[roundNumber].completed = true;
        
        // Apply reputation decay to all participants
        applyReputationDecay();
        
        // Increment current round
        currentRound++;
        
        emit AggregationCompleted(
            roundNumber,
            rounds[roundNumber].totalContributors,
            rounds[roundNumber].totalRewards,
            block.timestamp
        );
    }
    
    /**
     * @dev Calculate contribution score based on multiple factors
     * @param accuracyImprovement Accuracy improvement percentage
     * @param dataQuality Data quality score
     * @param reliability Model reliability score
     * @return contributionScore Calculated contribution score (0-100)
     */
    function calculateContributionScore(
        uint256 accuracyImprovement,
        uint256 dataQuality,
        uint256 reliability
    ) public pure returns (uint256) {
        uint256 weightedScore = (
            (accuracyImprovement * ACCURACY_WEIGHT) +
            (dataQuality * QUALITY_WEIGHT) +
            (reliability * RELIABILITY_WEIGHT)
        ) / 100;
        
        return weightedScore;
    }
    
    /**
     * @dev Calculate token reward based on contribution score and reputation
     * @param contributor Address of the contributor
     * @param contributionScore Contribution score (0-100)
     * @param verified Whether the contribution is verified
     * @return tokenReward Calculated token reward
     */
    function calculateTokenReward(
        address contributor,
        uint256 contributionScore,
        bool verified
    ) public view returns (uint256) {
        if (!verified) {
            return 0;
        }
        
        uint256 baseReward = (BASE_REWARD * contributionScore) / 100;
        uint256 reputationMultiplier = (participants[contributor].reputation * 2) / 1000; // 0-2x multiplier
        
        uint256 finalReward = (baseReward * (100 + reputationMultiplier)) / 100;
        
        return finalReward > MAX_REWARD ? MAX_REWARD : finalReward;
    }
    
    /**
     * @dev Update participant reputation based on contribution
     * @param contributor Address of the contributor
     * @param contributionScore Contribution score
     * @param verified Whether the contribution is verified
     */
    function updateReputation(
        address contributor,
        uint256 contributionScore,
        bool verified
    ) internal {
        uint256 oldReputation = participants[contributor].reputation;
        uint256 reputationChange = 0;
        
        if (verified) {
            // Positive reputation change based on contribution score
            reputationChange = (contributionScore * REPUTATION_BOOST) / 100;
            participants[contributor].verificationCount++;
        } else {
            // Negative reputation change for unverified contributions
            reputationChange = 20; // Fixed penalty
        }
        
        uint256 newReputation = oldReputation + reputationChange;
        
        // Ensure reputation stays within bounds (0-1000)
        if (newReputation > 1000) {
            newReputation = 1000;
        } else if (newReputation < 0) {
            newReputation = 0;
        }
        
        participants[contributor].reputation = newReputation;
        
        emit ReputationUpdated(
            contributor,
            oldReputation,
            newReputation,
            currentRound,
            block.timestamp
        );
    }
    
    /**
     * @dev Apply reputation decay to all participants
     */
    function applyReputationDecay() internal {
        // This would iterate through all participants and apply decay
        // For gas efficiency, this could be done off-chain or in batches
        // Implementation depends on the number of participants
    }
    
    /**
     * @dev Get participant information
     * @param participant Address of the participant
     * @return Participant information
     */
    function getParticipantInfo(address participant) external view returns (
        uint256 reputation,
        uint256 totalContributions,
        uint256 totalRewards,
        uint256 lastActivity,
        bool isActive,
        uint256 verificationCount
    ) {
        Participant memory p = participants[participant];
        return (
            p.reputation,
            p.totalContributions,
            p.totalRewards,
            p.lastActivity,
            p.isActive,
            p.verificationCount
        );
    }
    
    /**
     * @dev Get contribution information
     * @param roundNumber Round number
     * @param contributor Address of the contributor
     * @return Contribution information
     */
    function getContributionInfo(uint256 roundNumber, address contributor) external view returns (
        bytes32 modelHash,
        uint256 accuracyImprovement,
        uint256 dataQuality,
        uint256 reliability,
        uint256 contributionScore,
        uint256 tokenReward,
        bool verified,
        uint256 timestamp
    ) {
        ModelContribution memory c = contributions[roundNumber][contributor];
        return (
            c.modelHash,
            c.accuracyImprovement,
            c.dataQuality,
            c.reliability,
            c.contributionScore,
            c.tokenReward,
            c.verified,
            c.timestamp
        );
    }
    
    /**
     * @dev Get round information
     * @param roundNumber Round number
     * @return Round information
     */
    function getRoundInfo(uint256 roundNumber) external view returns (
        uint256 totalContributors,
        uint256 totalRewards,
        uint256 averageAccuracy,
        uint256 timestamp,
        bool completed
    ) {
        RoundData storage r = rounds[roundNumber];
        return (
            r.totalContributors,
            r.totalRewards,
            r.averageAccuracy,
            r.timestamp,
            r.completed
        );
    }
    
    /**
     * @dev Update aggregator address
     * @param newAggregator New aggregator address
     */
    function updateAggregator(address newAggregator) external onlyOwner {
        require(newAggregator != address(0), "Invalid aggregator address");
        aggregator = newAggregator;
    }
    
    /**
     * @dev Emergency function to pause the contract
     */
    function emergencyPause() external onlyOwner {
        // Implementation would depend on using OpenZeppelin's Pausable
        // This is a placeholder for emergency pause functionality
    }
}
