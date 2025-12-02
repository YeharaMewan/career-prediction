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
            "Hello! I'd love to learn more about you. What do you like to do in your free time?",
            "Welcome! Let's have a conversation and see where it takes us. What kind of things are you into these days?",
            "Hey there! I'm curious to hear about you. What's something you're really passionate about?",
            "Hi! Let's get to know each other. Tell me, what do you do for fun outside of school?",
            "Hi! I'm here to help you explore your future. Shall we start by getting to know a bit about you?",
            "Welcome! I'd love to chat and help you discover some exciting career paths. How are you doing today?",
        ]

        # Sinhala greetings (7 variations)
        self.pools["greetings_si"] = [
            "ආයුබෝවන්! ඔබව වඩාත් හොඳින් හඳුනා ගැනීමට මම මෙහි සිටිමි. ඔබ කැමති දේ ගැන කතා කිරීමෙන් පටන් ගමු ද?",
            "හෙලෝ! ඔබ ගැන තව දැනගන්න මම කැමතියි. ඔබේ විවේක වේලාවේ ඔබ මොනවාද කරන්න කැමති?",
            "සාදරයෙන් පිළිගනිමු! අපි කතා බහක් කරලා බලමු මොනවද වෙන්නේ කියලා. මේ දවස් වලදී ඔබ කැමති දේවල් මොනවාද?",
            "හායි! ඔබ ගැන ඇසීමට මට කුතුහලයි. ඔබ ඇත්තටම කැමති මොකක් කරන්නද?",
            "ආයුබෝවන්! අපි එකිනෙකා දැනගනිමු. කියන්න, පාසලෙන් පිටත විනෝදාංශ සඳහා ඔබ මොනව කරන්නද?",
            "ආයුබෝවන්! ඔබේ අනාගතය ගවේෂණය කිරීමට උදව් කිරීමට මම මෙතන ඉන්නවා. පළමුව ඔබ ගැන ටිකක් දැන ගෙන පටන් ගමු ද?",
            "සාදරයෙන් පිළිගනිමු!ආකර්ෂණීය වෘත්ති මාර්ග සොයා ගැනීමට උදව් කිරීමටයි කතා කිරීමටයි මම කැමතියි. අද ඔබට කොහොමද?",
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

        # Sinhala acknowledgments (20+ variations)
        self.pools["acknowledgments_si"] = [
            "ඒක හරිම අගනා අදහසක්!",
            "ඔබට ඒක ගැලපෙන්නේ ඇයිදයි මට තේරෙනවා.",
            "ඒක ඔබේ උනන්දුව පැහැදිලිව පෙන්නුම් කරනවා.",
            "රසවත් දෘෂ්ටිකෝණයක්!",
            "ඒක හරිම හොඳ නිරීක්ෂණයක්.",
            "ඔබ එය බෙදා ගැනීම ගැන මම අගය කරනවා.",
            "ඒක ඔබව මෙහෙයවන දේ ගැන මට බොහෝ දේ කියනවා.",
            "ඔබ වැදගත් දෙයක් අවබෝධ කර ගන්නවා.",
            "ඔබ ඒ ගැන සිතන අන්දම මට ඇත්තටම කැමතියි.",
            "ඒක බලන්න අපූරු ක්‍රමයක්.",
            "ඔබ මේක හොඳින් සිතා බලා ඇති.",
            "ඒක හරිම තේරෙනවා.",
            "ඔබ කිව්වේ මොකක්ද කියලා මට තේරෙනවා.",
            "ඒක හරිම වැදගත්.",
            "විශ්මයජනකයි!",
            "ඒක ඔබේ උනන්දුවන් ගැන බොහෝ දේ හෙළි කරනවා.",
            "ඔබ ඒ ගැන කතා කරන විට ඔබේ උනන්දුව මට ඇහෙනවා.",
            "ඒක හරිම පැහැදිලි නිරීක්ෂණයක්.",
            "ඔබ හොඳ දෙයකට එළඹෙනවා.",
            "ඔබ කියන දෙය තුළ ඒක දිලිහෙනවා.",
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
