"""
Peer-to-Peer Communication Network for Decentralized Federated Learning
Implements direct client-to-client communication without central servers
"""

import asyncio
import json
import logging
import socket
import threading
import time
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import struct

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of P2P messages"""
    HELLO = "hello"
    HELLO_RESPONSE = "hello_response"
    MODEL_UPDATE = "model_update"
    CONSENSUS_VOTE = "consensus_vote"
    AGGREGATION_RESULT = "aggregation_result"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    NODE_DISCOVERY = "node_discovery"
    HEARTBEAT = "heartbeat"
    PING = "ping"
    PONG = "pong"

class NodeStatus(Enum):
    """Node status in the network"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"

@dataclass
class PeerNode:
    """Represents a peer node in the network"""
    node_id: str
    address: str
    port: int
    status: NodeStatus
    last_seen: float
    connection_count: int = 0
    reliability_score: float = 1.0

@dataclass
class P2PMessage:
    """P2P message structure"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    round_number: int
    timestamp: float
    payload: Dict[str, Any]
    signature: Optional[str] = None

class P2PNetwork:
    """
    Peer-to-peer network for decentralized federated learning
    """
    
    def __init__(self, node_id: str, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize P2P network
        
        Args:
            node_id: Unique node identifier
            host: Host address
            port: Port number
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        
        # Network state
        self.peers: Dict[str, PeerNode] = {}
        self.connections: Dict[str, socket.socket] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        
        # Server components
        self.server_socket: Optional[socket.socket] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Message tracking
        self.sent_messages: Dict[str, P2PMessage] = {}
        self.received_messages: Dict[str, P2PMessage] = {}
        self.message_history: List[P2PMessage] = []
        
        # Network discovery
        self.bootstrap_nodes: List[Tuple[str, int]] = []
        self.discovery_interval = 30  # seconds
        
        # Initialize message handlers
        self._initialize_message_handlers()
        
        logger.info(f"P2P Network initialized: {node_id} at {host}:{port}")
    
    def start_network(self) -> bool:
        """
        Start the P2P network
        
        Returns:
            success: Whether network started successfully
        """
        try:
            # Start server
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.is_running = True
            
            logger.info(f"P2P Network started: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start P2P network: {str(e)}")
            return False
    
    def stop_network(self):
        """
        Stop the P2P network
        """
        self.is_running = False
        
        # Close all connections
        for peer_id, connection in self.connections.items():
            try:
                connection.close()
            except:
                pass
        
        self.connections.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Wait for server thread
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        logger.info("P2P Network stopped")
    
    def connect_to_peer(self, peer_host: str, peer_port: int) -> bool:
        """
        Connect to a peer node
        
        Args:
            peer_host: Peer host address
            peer_port: Peer port number
            
        Returns:
            success: Whether connection was successful
        """
        try:
            peer_address = f"{peer_host}:{peer_port}"
            
            # Create connection
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.connect((peer_host, peer_port))
            
            # Send hello message
            hello_message = P2PMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.HELLO,
                sender_id=self.node_id,
                recipient_id=None,
                round_number=0,
                timestamp=time.time(),
                payload={
                    'node_id': self.node_id,
                    'host': self.host,
                    'port': self.port
                }
            )
            
            self._send_message(connection, hello_message)
            
            # Wait for response
            response = self._receive_message(connection)
            if response and response.message_type == MessageType.HELLO_RESPONSE:
                peer_id = response.payload.get('node_id')
                if peer_id:
                    # Add peer
                    peer_node = PeerNode(
                        node_id=peer_id,
                        address=peer_host,
                        port=peer_port,
                        status=NodeStatus.CONNECTED,
                        last_seen=time.time()
                    )
                    
                    self.peers[peer_id] = peer_node
                    self.connections[peer_id] = connection
                    
                    logger.info(f"Connected to peer: {peer_id} at {peer_address}")
                    return True
            
            connection.close()
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_host}:{peer_port}: {str(e)}")
            return False
    
    def broadcast_message(self, message_type: MessageType, payload: Dict[str, Any], 
                         round_number: int = 0) -> List[str]:
        """
        Broadcast message to all connected peers
        
        Args:
            message_type: Type of message
            payload: Message payload
            round_number: Round number
            
        Returns:
            sent_to: List of peer IDs that received the message
        """
        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.node_id,
            recipient_id=None,  # Broadcast
            round_number=round_number,
            timestamp=time.time(),
            payload=payload
        )
        
        sent_to = []
        
        for peer_id, connection in list(self.connections.items()):
            try:
                if self._send_message(connection, message):
                    sent_to.append(peer_id)
                    self.peers[peer_id].last_seen = time.time()
            except Exception as e:
                logger.error(f"Failed to send message to peer {peer_id}: {str(e)}")
                # Mark peer as disconnected
                self.peers[peer_id].status = NodeStatus.DISCONNECTED
        
        # Store sent message
        self.sent_messages[message.message_id] = message
        
        logger.info(f"Broadcasted {message_type.value} to {len(sent_to)} peers")
        return sent_to
    
    def send_message_to_peer(self, peer_id: str, message_type: MessageType, 
                           payload: Dict[str, Any], round_number: int = 0) -> bool:
        """
        Send message to specific peer
        
        Args:
            peer_id: Peer identifier
            message_type: Type of message
            payload: Message payload
            round_number: Round number
            
        Returns:
            success: Whether message was sent successfully
        """
        if peer_id not in self.connections:
            logger.error(f"No connection to peer {peer_id}")
            return False
        
        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.node_id,
            recipient_id=peer_id,
            round_number=round_number,
            timestamp=time.time(),
            payload=payload
        )
        
        try:
            connection = self.connections[peer_id]
            success = self._send_message(connection, message)
            
            if success:
                self.peers[peer_id].last_seen = time.time()
                self.sent_messages[message.message_id] = message
                logger.info(f"Sent {message_type.value} to peer {peer_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message to peer {peer_id}: {str(e)}")
            return False
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register message handler for specific message type
        
        Args:
            message_type: Type of message
            handler: Handler function
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for {message_type.value}")
    
    def get_network_status(self) -> Dict:
        """
        Get network status information
        
        Returns:
            status: Network status
        """
        connected_peers = [p for p in self.peers.values() if p.status == NodeStatus.CONNECTED]
        
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'is_running': self.is_running,
            'total_peers': len(self.peers),
            'connected_peers': len(connected_peers),
            'active_connections': len(self.connections),
            'messages_sent': len(self.sent_messages),
            'messages_received': len(self.received_messages),
            'peer_list': [
                {
                    'node_id': peer.node_id,
                    'address': f"{peer.address}:{peer.port}",
                    'status': peer.status.value,
                    'last_seen': peer.last_seen,
                    'reliability_score': peer.reliability_score
                }
                for peer in self.peers.values()
            ]
        }
    
    def discover_peers(self, bootstrap_nodes: List[Tuple[str, int]]) -> int:
        """
        Discover peers using bootstrap nodes
        
        Args:
            bootstrap_nodes: List of bootstrap node addresses
            
        Returns:
            discovered_count: Number of peers discovered
        """
        discovered_count = 0
        
        for host, port in bootstrap_nodes:
            try:
                # Connect to bootstrap node
                if self.connect_to_peer(host, port):
                    # Request peer list
                    peer_list_message = P2PMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.NODE_DISCOVERY,
                        sender_id=self.node_id,
                        recipient_id=None,
                        round_number=0,
                        timestamp=time.time(),
                        payload={'request': 'peer_list'}
                    )
                    
                    # Send discovery request
                    bootstrap_peer_id = None
                    for pid, conn in self.connections.items():
                        if conn.getpeername()[0] == host and conn.getpeername()[1] == port:
                            bootstrap_peer_id = pid
                            break
                    
                    if bootstrap_peer_id:
                        self.send_message_to_peer(bootstrap_peer_id, MessageType.NODE_DISCOVERY, 
                                                {'request': 'peer_list'})
                        discovered_count += 1
                
            except Exception as e:
                logger.error(f"Failed to discover peers from {host}:{port}: {str(e)}")
        
        logger.info(f"Discovered {discovered_count} peers from bootstrap nodes")
        return discovered_count
    
    def _server_loop(self):
        """
        Server loop to accept incoming connections
        """
        logger.info("Started P2P server loop")
        
        while self.is_running:
            try:
                # Accept connection
                connection, address = self.server_socket.accept()
                
                # Handle connection in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client_connection,
                    args=(connection, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error in server loop: {str(e)}")
                time.sleep(1.0)
        
        logger.info("P2P server loop stopped")
    
    def _handle_client_connection(self, connection: socket.socket, address: Tuple[str, int]):
        """
        Handle incoming client connection
        
        Args:
            connection: Client connection socket
            address: Client address
        """
        peer_id = None
        
        try:
            # Receive hello message
            hello_message = self._receive_message(connection)
            if not hello_message or hello_message.message_type != MessageType.HELLO:
                connection.close()
                return
            
            peer_id = hello_message.payload.get('node_id')
            peer_host = hello_message.payload.get('host')
            peer_port = hello_message.payload.get('port')
            
            if not peer_id:
                connection.close()
                return
            
            # Send hello response
            response = P2PMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.HELLO_RESPONSE,
                sender_id=self.node_id,
                recipient_id=peer_id,
                round_number=0,
                timestamp=time.time(),
                payload={
                    'node_id': self.node_id,
                    'host': self.host,
                    'port': self.port,
                    'accepted': True
                }
            )
            
            self._send_message(connection, response)
            
            # Add peer
            peer_node = PeerNode(
                node_id=peer_id,
                address=peer_host,
                port=peer_port,
                status=NodeStatus.CONNECTED,
                last_seen=time.time()
            )
            
            self.peers[peer_id] = peer_node
            self.connections[peer_id] = connection
            
            logger.info(f"Accepted connection from peer: {peer_id} at {address}")
            
            # Handle incoming messages
            while self.is_running:
                try:
                    message = self._receive_message(connection)
                    if message:
                        self._handle_received_message(message)
                    else:
                        break
                except Exception as e:
                    logger.error(f"Error handling message from {peer_id}: {str(e)}")
                    break
        
        except Exception as e:
            logger.error(f"Error handling client connection from {address}: {str(e)}")
        
        finally:
            # Cleanup connection
            if peer_id:
                if peer_id in self.connections:
                    del self.connections[peer_id]
                if peer_id in self.peers:
                    self.peers[peer_id].status = NodeStatus.DISCONNECTED
            
            try:
                connection.close()
            except:
                pass
            
            if peer_id:
                logger.info(f"Disconnected peer: {peer_id}")
    
    def _handle_received_message(self, message: P2PMessage):
        """
        Handle received message
        
        Args:
            message: Received message
        """
        # Store received message
        self.received_messages[message.message_id] = message
        self.message_history.append(message)
        
        # Keep only recent message history
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-500:]
        
        # Update peer status
        if message.sender_id in self.peers:
            self.peers[message.sender_id].last_seen = time.time()
        
        # Call registered handlers
        if message.message_type in self.message_handlers:
            for handler in self.message_handlers[message.message_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {str(e)}")
        
        logger.debug(f"Handled message: {message.message_type.value} from {message.sender_id}")
    
    def _send_message(self, connection: socket.socket, message: P2PMessage) -> bool:
        """
        Send message through connection
        
        Args:
            connection: Connection socket
            message: Message to send
            
        Returns:
            success: Whether message was sent successfully
        """
        try:
            # Serialize message
            message_data = json.dumps(asdict(message), default=str).encode('utf-8')
            message_length = len(message_data)
            
            # Send message length first
            connection.send(struct.pack('!I', message_length))
            
            # Send message data
            connection.send(message_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False
    
    def _receive_message(self, connection: socket.socket) -> Optional[P2PMessage]:
        """
        Receive message from connection
        
        Args:
            connection: Connection socket
            
        Returns:
            message: Received message or None
        """
        try:
            # Receive message length
            length_data = connection.recv(4)
            if len(length_data) < 4:
                return None
            
            message_length = struct.unpack('!I', length_data)[0]
            
            # Receive message data
            message_data = b''
            while len(message_data) < message_length:
                chunk = connection.recv(message_length - len(message_data))
                if not chunk:
                    return None
                message_data += chunk
            
            # Deserialize message
            message_dict = json.loads(message_data.decode('utf-8'))
            
            # Convert to P2PMessage
            message = P2PMessage(
                message_id=message_dict['message_id'],
                message_type=MessageType(message_dict['message_type']),
                sender_id=message_dict['sender_id'],
                recipient_id=message_dict['recipient_id'],
                round_number=message_dict['round_number'],
                timestamp=message_dict['timestamp'],
                payload=message_dict['payload'],
                signature=message_dict.get('signature')
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive message: {str(e)}")
            return None
    
    def _initialize_message_handlers(self):
        """
        Initialize default message handlers
        """
        def handle_hello(message: P2PMessage):
            logger.info(f"Received hello from {message.sender_id}")
        
        def handle_hello_response(message: P2PMessage):
            logger.info(f"Received hello response from {message.sender_id}")
        
        def handle_heartbeat(message: P2PMessage):
            # Update peer last seen time
            if message.sender_id in self.peers:
                self.peers[message.sender_id].last_seen = time.time()
        
        def handle_ping(message: P2PMessage):
            # Respond with pong
            self.send_message_to_peer(message.sender_id, MessageType.PONG, 
                                    {'pong_id': message.payload.get('ping_id')})
        
        self.register_message_handler(MessageType.HELLO, handle_hello)
        self.register_message_handler(MessageType.HELLO_RESPONSE, handle_hello_response)
        self.register_message_handler(MessageType.HEARTBEAT, handle_heartbeat)
        self.register_message_handler(MessageType.PING, handle_ping)

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing P2P Network")
    
    # Create P2P network
    network = P2PNetwork("node_1", "127.0.0.1", 8001)
    
    # Register custom message handlers
    def handle_model_update(message: P2PMessage):
        logger.info(f"Received model update from {message.sender_id}: {message.payload}")
    
    def handle_consensus_vote(message: P2PMessage):
        logger.info(f"Received consensus vote from {message.sender_id}: {message.payload}")
    
    network.register_message_handler(MessageType.MODEL_UPDATE, handle_model_update)
    network.register_message_handler(MessageType.CONSENSUS_VOTE, handle_consensus_vote)
    
    # Start network
    if network.start_network():
        logger.info("P2P Network started successfully")
        
        # Connect to bootstrap nodes (if any)
        bootstrap_nodes = [("127.0.0.1", 8002), ("127.0.0.1", 8003)]
        network.discover_peers(bootstrap_nodes)
        
        # Send test messages
        time.sleep(2)
        
        # Broadcast model update
        network.broadcast_message(
            MessageType.MODEL_UPDATE,
            {
                'model_hash': 'abc123',
                'round_number': 1,
                'parameters': {'layer1.weight': [0.1, 0.2, 0.3]}
            },
            round_number=1
        )
        
        # Send consensus vote
        network.broadcast_message(
            MessageType.CONSENSUS_VOTE,
            {
                'model_hash': 'abc123',
                'vote_weight': 100.0,
                'round_number': 1
            },
            round_number=1
        )
        
        # Check network status
        time.sleep(5)
        status = network.get_network_status()
        logger.info(f"Network status: {json.dumps(status, indent=2)}")
        
        # Stop network
        network.stop_network()
    
    logger.info("✅ P2P Network test completed")
