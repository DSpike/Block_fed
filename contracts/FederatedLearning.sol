// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FederatedLearning
 * @dev Smart contract for blockchain-based federated learning
 * @author Your Name
 */
contract FederatedLearning {
    
    // Structs
    struct ModelUpdate {
        string participantId;
        bytes32 modelHash;
        uint256 qualityScore;
        uint256 timestamp;
        bool isValid;
    }
    
    struct Participant {
        address participantAddress;
        uint256 totalIncentives;
        uint256 participationCount;
        bool isRegistered;
    }
    
    // State variables
    mapping(string => ModelUpdate) public modelUpdates;
    mapping(string => Participant) public participants;
    mapping(uint256 => bytes32) public aggregatedModels;
    
    uint256 public roundNumber;
    uint256 public minParticipants;
    uint256 public incentivePool;
    address public owner;
    
    // Events
    event ModelSubmitted(string participantId, bytes32 modelHash, uint256 qualityScore);
    event ModelsAggregated(uint256 roundNumber, bytes32 aggregatedHash);
    event IncentiveAwarded(string participantId, uint256 amount);
    event ParticipantRegistered(string participantId, address participantAddress);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyRegisteredParticipant(string memory participantId) {
        require(participants[participantId].isRegistered, "Participant not registered");
        _;
    }
    
    // Constructor
    constructor(uint256 _minParticipants) {
        owner = msg.sender;
        minParticipants = _minParticipants;
        roundNumber = 0;
        incentivePool = 0;
    }
    
    /**
     * @dev Register a new participant
     * @param participantId Unique identifier for the participant
     */
    function registerParticipant(string memory participantId) public {
        require(!participants[participantId].isRegistered, "Participant already registered");
        
        participants[participantId] = Participant({
            participantAddress: msg.sender,
            totalIncentives: 0,
            participationCount: 0,
            isRegistered: true
        });
        
        emit ParticipantRegistered(participantId, msg.sender);
    }
    
    /**
     * @dev Submit model update to the blockchain
     * @param participantId Unique identifier for the participant
     * @param modelHash Hash of the model parameters
     * @param qualityScore Quality score of the model (0-1000)
     */
    function submitModelUpdate(
        string memory participantId,
        bytes32 modelHash,
        uint256 qualityScore
    ) public onlyRegisteredParticipant(participantId) {
        require(qualityScore <= 1000, "Quality score must be between 0 and 1000");
        require(!modelUpdates[participantId].isValid, "Model already submitted for this round");
        
        // Create model update
        modelUpdates[participantId] = ModelUpdate({
            participantId: participantId,
            modelHash: modelHash,
            qualityScore: qualityScore,
            timestamp: block.timestamp,
            isValid: true
        });
        
        // Update participant stats
        participants[participantId].participationCount++;
        
        emit ModelSubmitted(participantId, modelHash, qualityScore);
    }
    
    /**
     * @dev Aggregate models and award incentives
     * @return aggregatedHash Hash of the aggregated model
     */
    function aggregateModels() public onlyOwner returns (bytes32) {
        require(getValidSubmissionCount() >= minParticipants, "Insufficient participants");
        
        // Calculate total quality score
        uint256 totalQuality = 0;
        uint256 participantCount = 0;
        
        // Count participants and calculate total quality
        for (uint256 i = 0; i < 100; i++) { // Limit to prevent infinite loop
            string memory participantId = string(abi.encodePacked("client_", uint2str(i)));
            if (modelUpdates[participantId].isValid) {
                totalQuality += modelUpdates[participantId].qualityScore;
                participantCount++;
            }
        }
        
        require(participantCount >= minParticipants, "Not enough valid submissions");
        
        // Award incentives based on quality
        for (uint256 i = 0; i < 100; i++) {
            string memory participantId = string(abi.encodePacked("client_", uint2str(i)));
            if (modelUpdates[participantId].isValid) {
                uint256 incentive = (modelUpdates[participantId].qualityScore * 10) / totalQuality;
                participants[participantId].totalIncentives += incentive;
                incentivePool += incentive;
                
                emit IncentiveAwarded(participantId, incentive);
            }
        }
        
        // Create aggregated model hash (simplified for demonstration)
        bytes32 aggregatedHash = keccak256(abi.encodePacked(
            roundNumber,
            block.timestamp,
            totalQuality
        ));
        
        aggregatedModels[roundNumber] = aggregatedHash;
        roundNumber++;
        
        // Clear model updates for next round
        clearModelUpdates();
        
        emit ModelsAggregated(roundNumber - 1, aggregatedHash);
        
        return aggregatedHash;
    }
    
    /**
     * @dev Get incentive balance for a participant
     * @param participantId Unique identifier for the participant
     * @return Total incentives earned
     */
    function getIncentive(string memory participantId) public view returns (uint256) {
        return participants[participantId].totalIncentives;
    }
    
    /**
     * @dev Get current round number
     * @return Current round number
     */
    function getRoundNumber() public view returns (uint256) {
        return roundNumber;
    }
    
    /**
     * @dev Get number of valid submissions
     * @return Count of valid submissions
     */
    function getValidSubmissionCount() public view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < 100; i++) {
            string memory participantId = string(abi.encodePacked("client_", uint2str(i)));
            if (modelUpdates[participantId].isValid) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * @dev Clear model updates for next round
     */
    function clearModelUpdates() internal {
        for (uint256 i = 0; i < 100; i++) {
            string memory participantId = string(abi.encodePacked("client_", uint2str(i)));
            if (modelUpdates[participantId].isValid) {
                modelUpdates[participantId].isValid = false;
            }
        }
    }
    
    /**
     * @dev Convert uint to string (helper function)
     * @param _i Number to convert
     * @return String representation
     */
    function uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 length;
        while (j != 0) {
            length++;
            j /= 10;
        }
        bytes memory bstr = new bytes(length);
        uint256 k = length;
        while (_i != 0) {
            k -= 1;
            uint8 temp = (48 + uint8(_i - _i / 10 * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }
    
    /**
     * @dev Add funds to incentive pool (only owner)
     */
    function addIncentivePool() public payable onlyOwner {
        incentivePool += msg.value;
    }
    
    /**
     * @dev Withdraw incentives (participants can withdraw their earned incentives)
     * @param participantId Unique identifier for the participant
     */
    function withdrawIncentives(string memory participantId) public {
        require(participants[participantId].isRegistered, "Participant not registered");
        require(participants[participantId].participantAddress == msg.sender, "Not authorized");
        
        uint256 amount = participants[participantId].totalIncentives;
        require(amount > 0, "No incentives to withdraw");
        
        participants[participantId].totalIncentives = 0;
        
        payable(msg.sender).transfer(amount);
    }
    
    /**
     * @dev Get contract balance
     * @return Contract balance in wei
     */
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}






