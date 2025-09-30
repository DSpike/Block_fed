// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FederatedLearningToken (FLT)
 * @dev ERC20 token for federated learning incentive rewards
 * @author Federated Learning System
 */
contract FederatedLearningToken {
    string public name = "Federated Learning Token";
    string public symbol = "FLT";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    address public owner;
    address public incentiveContract;
    
    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyIncentiveContract() {
        require(msg.sender == incentiveContract, "Only incentive contract can call this function");
        _;
    }
    
    constructor(uint256 _initialSupply) {
        owner = msg.sender;
        totalSupply = _initialSupply * 10**decimals;
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }
    
    /**
     * @dev Set the incentive contract address
     * @param _incentiveContract Address of the incentive contract
     */
    function setIncentiveContract(address _incentiveContract) external onlyOwner {
        incentiveContract = _incentiveContract;
    }
    
    /**
     * @dev Transfer tokens to a recipient
     * @param to Recipient address
     * @param value Amount to transfer
     * @return success Whether transfer was successful
     */
    function transfer(address to, uint256 value) external returns (bool success) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        require(to != address(0), "Transfer to zero address");
        
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    /**
     * @dev Transfer tokens from one address to another
     * @param from Sender address
     * @param to Recipient address
     * @param value Amount to transfer
     * @return success Whether transfer was successful
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool success) {
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");
        require(to != address(0), "Transfer to zero address");
        
        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        
        emit Transfer(from, to, value);
        return true;
    }
    
    /**
     * @dev Approve spender to transfer tokens
     * @param spender Address to approve
     * @param value Amount to approve
     * @return success Whether approval was successful
     */
    function approve(address spender, uint256 value) external returns (bool success) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }
    
    /**
     * @dev Mint new tokens (only incentive contract can call)
     * @param to Address to mint tokens to
     * @param amount Amount to mint
     */
    function mint(address to, uint256 amount) external onlyIncentiveContract {
        require(to != address(0), "Mint to zero address");
        
        totalSupply += amount;
        balanceOf[to] += amount;
        
        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
    }
    
    /**
     * @dev Burn tokens (only incentive contract can call)
     * @param from Address to burn tokens from
     * @param amount Amount to burn
     */
    function burn(address from, uint256 amount) external onlyIncentiveContract {
        require(balanceOf[from] >= amount, "Insufficient balance to burn");
        
        balanceOf[from] -= amount;
        totalSupply -= amount;
        
        emit Burn(from, amount);
        emit Transfer(from, address(0), amount);
    }
    
    /**
     * @dev Distribute rewards to multiple recipients
     * @param recipients Array of recipient addresses
     * @param amounts Array of amounts to distribute
     */
    function distributeRewards(address[] calldata recipients, uint256[] calldata amounts) external onlyIncentiveContract {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length > 0, "No recipients");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            address recipient = recipients[i];
            uint256 amount = amounts[i];
            
            require(recipient != address(0), "Transfer to zero address");
            require(balanceOf[msg.sender] >= amount, "Insufficient balance for distribution");
            
            balanceOf[msg.sender] -= amount;
            balanceOf[recipient] += amount;
            
            emit Transfer(msg.sender, recipient, amount);
        }
    }
    
    /**
     * @dev Get token balance of an address
     * @param account Address to check balance
     * @return balance Token balance
     */
    function getBalance(address account) external view returns (uint256 balance) {
        return balanceOf[account];
    }
    
    /**
     * @dev Get total supply
     * @return supply Total token supply
     */
    function getTotalSupply() external view returns (uint256 supply) {
        return totalSupply;
    }
}

