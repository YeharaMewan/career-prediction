"""
LangSmith Configuration and Setup for Career Planning System

This module handles LangSmith initialization, tracing configuration,
and monitoring setup for all agents and workflows.
"""
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LangSmithConfig:
    """Configuration manager for LangSmith integration"""
    
    def __init__(self):
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "career-planning-system")
        self.tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
    def setup_langsmith(self) -> bool:
        """
        Setup LangSmith for the current session
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        if not self.api_key:
            logger.warning("LANGSMITH_API_KEY not found. LangSmith monitoring disabled.")
            return False
            
        try:
            # Set environment variables for LangChain to pick up
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.tracing_enabled).lower()
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
            
            if self.tracing_enabled:
                logger.info(f"LangSmith tracing enabled for project: {self.project_name}")
                logger.info(f"LangSmith endpoint: {self.endpoint}")
            else:
                logger.info("LangSmith configured but tracing disabled")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup LangSmith: {e}")
            return False
    
    def get_run_config(self, tags: Optional[list] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get configuration for LangSmith run tracking
        
        Args:
            tags: Optional list of tags for the run
            metadata: Optional metadata dictionary
            
        Returns:
            Dict containing run configuration
        """
        config = {
            "configurable": {
                "thread_id": f"career-session-{hash(str(metadata) if metadata else 'default')}",
            }
        }
        
        if self.tracing_enabled and self.api_key:
            run_config = {}
            
            if tags:
                run_config["tags"] = tags
                
            if metadata:
                run_config["metadata"] = metadata
                
            if run_config:
                config["langsmith"] = run_config
                
        return config
    
    def create_session_tags(self, session_type: str, agent_name: str) -> list:
        """
        Create standardized tags for LangSmith runs
        
        Args:
            session_type: Type of session (e.g., 'career_prediction', 'profiling')
            agent_name: Name of the agent handling the session
            
        Returns:
            List of tags
        """
        return [
            f"session_type:{session_type}",
            f"agent:{agent_name}",
            f"project:career-planning",
            f"environment:{'dev' if os.getenv('DEBUG', 'false').lower() == 'true' else 'prod'}"
        ]
    
    def create_session_metadata(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create standardized metadata for LangSmith runs
        
        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "project": "career-planning-system",
            "version": "1.0.0",
            "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        }
        
        if user_id:
            metadata["user_id"] = user_id
            
        if session_id:
            metadata["session_id"] = session_id
            
        return metadata


# Global instance
langsmith_config = LangSmithConfig()

def setup_langsmith() -> bool:
    """
    Setup LangSmith for the application
    
    Returns:
        bool: True if setup successful
    """
    return langsmith_config.setup_langsmith()

def get_traced_run_config(
    session_type: str = "career_planning",
    agent_name: str = "main_supervisor", 
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    additional_tags: Optional[list] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get a complete run configuration with tracing enabled
    
    Args:
        session_type: Type of session
        agent_name: Name of the agent
        user_id: Optional user ID
        session_id: Optional session ID
        additional_tags: Additional tags to include
        additional_metadata: Additional metadata to include
        
    Returns:
        Complete run configuration
    """
    tags = langsmith_config.create_session_tags(session_type, agent_name)
    if additional_tags:
        tags.extend(additional_tags)
    
    metadata = langsmith_config.create_session_metadata(user_id, session_id)
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return langsmith_config.get_run_config(tags, metadata)

def log_agent_execution(agent_name: str, action: str, result: str, duration: float = None):
    """
    Log agent execution details to both logging and LangSmith
    
    Args:
        agent_name: Name of the executing agent
        action: Action being performed
        result: Result or status
        duration: Optional execution duration in seconds
    """
    log_msg = f"Agent {agent_name} - {action}: {result}"
    if duration:
        log_msg += f" (Duration: {duration:.2f}s)"
    
    logger.info(log_msg)
    
    # If LangSmith is enabled, this will be automatically captured
    # through the tracing system

# Initialize LangSmith on module import
if __name__ != "__main__":
    setup_success = setup_langsmith()
    if setup_success:
        logger.info("LangSmith configuration loaded successfully")
    else:
        logger.warning("LangSmith configuration failed or disabled")