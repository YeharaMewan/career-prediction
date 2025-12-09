"""
Pre-generated message pools for random selection.
Reduces LLM costs while maintaining variety.

This module provides the MessagePool class which manages pools of pre-generated
messages (greetings, acknowledgments, etc.) that can be randomly selected
instead of generating fresh messages every time.
"""

from typing import Dict, List
from datetime import datetime, timedelta
import random
import logging


class MessagePool:
    """
    Manages pools of pre-generated messages for random selection.

    Provides different message types (greetings, acknowledgments) in multiple
    languages (English, Sinhala) with automatic pool refresh capabilities.
    """

    def __init__(self):
        """Initialize message pools with empty dictionaries."""
        self.logger = logging.getLogger(__name__)

        # Initialize all pools
        self.pools = {
            "greetings_en": [],
            "greetings_si": [],
            "acknowledgments_en": [],
            "acknowledgments_si": [],
        }

        # Tracking for pool refresh
        self.last_refresh = None
        self.refresh_interval = timedelta(days=7)  # Refresh weekly

        # Initialize pools with default content
        self._init_default_pools()

        self.logger.info("✅ MessagePool initialized with pre-generated content")

    def _init_default_pools(self):
        """Initialize all pools with default pre-generated content."""
        self._init_greeting_pools()
        self._init_acknowledgment_pools()
        self.last_refresh = datetime.now()

    def _init_greeting_pools(self):
        """Initialize greeting pools for both languages."""

        # English greetings (7 variations)
        self.pools["greetings_en"] = [
            "Hi! I'm here to chat and get to know you better. Shall we start by talking about what you enjoy doing?",
            "Hello! I'd love to learn more about you. What are some things that really excite or interest you?",
            "Welcome! Let's have a conversation and see where it takes us. What kind of things are you into these days?",
            "Hey there! I'm curious to hear about you. What's something you're really passionate about?",
            "Hi! Let's get to know each other. What are some of your favorite hobbies or interests?",
            "Hi! I'm here to help you explore your future. Shall we start by getting to know a bit about you?",
            "Welcome! I'd love to chat and help you discover some exciting career paths. How are you doing today?",
        ]

        # Sinhala greetings (7 variations) - Natural conversational style
        self.pools["greetings_si"] = [
            "ආයුබෝවන්! ඔයා ගැන හොඳට දැනගන්න කතා කරමු. ඔයා කැමති දේවල් ගැන කතා කරන්න පටන් ගමුද?",
            "හෙලෝ! ඔයා ගැන තව දැනගන්න කැමතියි. ඔයාට real interested වෙන හෝ exciting ලගන කරන දේවල් මොනවාද?",
            "සාදරයෙන් පිළිගනිමු! අපි chat එකක් කරලා බලමු. මේ දවස්වල ඔයා වැඩිම enjoy කරන දේවල් මොනවාද?",
            "හායි! ඔයා ගැන දැනගන්න curious වෙලා ඉන්නවා. ඔයා කැමති මොනවද කරන්න?",
            "ආයුබෝවන්! අපි එකිනෙක දැනගනිමු. කියන්න, ඔයාගේ favorite hobbies හෝ interests මොනවාද?",
            "ආයුබෝවන්! ඔයාගේ future explore කරන්න help කරන්න මෙතන ඉන්නවා. පළමුව ඔයා ගැන ටිකක් දැනගන්න පටන් ගමුද?",
            "සාදරයෙන් පිළිගනිමු! Career pathways සොයාගන්න help කරන්න කතා කරන්න කැමතියි. අද ඔයාට කොහොමද?",
        ]

        self.logger.info(
            f"Initialized greeting pools: {len(self.pools['greetings_en'])} EN, {len(self.pools['greetings_si'])} SI"
        )

    def _init_acknowledgment_pools(self):
        """Initialize acknowledgment pools for both languages."""

        # English acknowledgments (20+ variations for variety)
        self.pools["acknowledgments_en"] = [
            "That's insightful!",
            "I can see why that resonates with you.",
            "That really shows your interests.",
            "Interesting perspective!",
            "That's a great observation.",
            "I appreciate you sharing that.",
            "That tells me a lot about what drives you.",
            "You're picking up on something important there.",
            "I really like how you think about that.",
            "That's a wonderful way to look at it.",
            "You've clearly thought this through.",
            "That makes a lot of sense.",
            "I see what you mean.",
            "That's quite telling.",
            "Fascinating!",
            "That's very revealing about your interests.",
            "I can hear your passion when you talk about that.",
            "That's a keen observation.",
            "You're onto something there.",
            "That shines through in what you're saying.",
        ]

        # Sinhala acknowledgments (20+ variations) - Natural conversational style
        self.pools["acknowledgments_si"] = [
            "නියමයි!",
            "ඔයාට ඒක resonate වෙන්නේ ඇයිද මට තේරෙනවා.",
            "ඒකෙන් ඔයාගේ interests පේනවා.",
            "Interesting perspective එකක්!",
            "ඒක හොඳ observation එකක්.",
            "ඔයා share කරපු එකට thanks!",
            "ඔයා වැදගත් දෙයක් අවබෝධ කරගන්නවා.",
            "ඔයා ඒ ගැන think කරන විදිය really nice.",
            "ඒක බලන විදිය හරිම හොඳයි.",
            "ඔයා මේක හොඳට plan කරලා ඉන්නවා පේනවා.",
            "ඔයා කිව්වේ මොකද කියලා මට clear.",
            "ඒක quite telling එකක්.",
            "Fascinating!",
            "ඒකෙන් ඔයාගේ passion එක පේනවා.",
            "ඔයා ඒ ගැන කතා කරද්දී interest එක පේනවා.",
            "ඒක keen observation එකක්.",
            "ඔයා හොඳ දෙයකට පැමිණෙනවා.",
            "ඔයා කියන එකෙන් ඒක හොඳට shine වෙනවා.",
            "හරිම sense make වෙනවා.",
            "ඔයාගේ passion එක hear කරන්න පුළුවන් ඒකෙන්.",
        ]

        self.logger.info(
            f"Initialized acknowledgment pools: {len(self.pools['acknowledgments_en'])} EN, {len(self.pools['acknowledgments_si'])} SI"
        )

    def get_random_greeting(self, language: str = "en") -> str:
        """
        Get a random greeting from the pool for the specified language.

        Args:
            language: Language code ("en" or "si")

        Returns:
            Random greeting message

        Raises:
            ValueError: If language is not supported
        """
        if language not in ["en", "si"]:
            self.logger.warning(
                f"Unsupported language '{language}', defaulting to English"
            )
            language = "en"

        pool_key = f"greetings_{language}"

        # Check if pool needs refresh (older than 7 days)
        if self._needs_refresh():
            self.logger.info("Message pools are stale, refreshing...")
            self._init_default_pools()

        if not self.pools[pool_key]:
            self.logger.error(f"Pool '{pool_key}' is empty! Reinitializing...")
            self._init_greeting_pools()

        greeting = random.choice(self.pools[pool_key])
        self.logger.debug(f"Selected random greeting from {pool_key} pool")

        return greeting

    def get_random_acknowledgment(self, language: str = "en") -> str:
        """
        Get a random acknowledgment phrase from the pool.

        Args:
            language: Language code ("en" or "si")

        Returns:
            Random acknowledgment phrase like "Great point!", "I see!"
        """
        if language not in ["en", "si"]:
            self.logger.warning(
                f"Unsupported language '{language}', defaulting to English"
            )
            language = "en"

        pool_key = f"acknowledgments_{language}"

        if not self.pools[pool_key]:
            self.logger.error(f"Pool '{pool_key}' is empty! Reinitializing...")
            self._init_acknowledgment_pools()

        acknowledgment = random.choice(self.pools[pool_key])
        self.logger.debug(f"Selected random acknowledgment from {pool_key} pool")

        return acknowledgment

    def _needs_refresh(self) -> bool:
        """
        Check if pools need to be refreshed based on time elapsed.

        Returns:
            True if pools should be refreshed, False otherwise
        """
        if self.last_refresh is None:
            return True

        elapsed = datetime.now() - self.last_refresh
        return elapsed > self.refresh_interval

    def add_greeting(self, language: str, greeting: str):
        """
        Add a new greeting to the pool (useful for LLM-generated greetings).

        Args:
            language: Language code ("en" or "si")
            greeting: The greeting text to add
        """
        pool_key = f"greetings_{language}"

        if pool_key in self.pools:
            # Avoid duplicates
            if greeting not in self.pools[pool_key]:
                self.pools[pool_key].append(greeting)
                self.logger.info(
                    f"Added new greeting to {pool_key} pool (total: {len(self.pools[pool_key])})"
                )
        else:
            self.logger.error(f"Unknown pool key: {pool_key}")

    def get_pool_stats(self) -> Dict[str, int]:
        """
        Get statistics about current pool sizes.

        Returns:
            Dictionary with pool names and their sizes
        """
        return {pool_name: len(messages) for pool_name, messages in self.pools.items()}
