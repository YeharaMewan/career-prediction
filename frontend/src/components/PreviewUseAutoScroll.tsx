import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { autoScrollListRef } from "../hooks/use-auto-scroll";
import { useNavigate } from "react-router-dom";
import apiService from "../services/api";
import type { Message } from "../types/chat";

const PreviewUseAutoScroll = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCompleted, setIsCompleted] = useState(false);
  const navigate = useNavigate();
  const userMessagesCount = messages.filter((msg) => msg.sender === "user").length;

  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize session when component mounts
  useEffect(() => {
    const initializeChat = async () => {
      try {
        setIsLoading(true);
        const response = await apiService.initializeSession();
        setSessionId(response.session_id);

        // Add the agent's initial question
        if (response.question) {
          setMessages([{ sender: "ai", text: response.question }]);
        } else {
          setMessages([{ sender: "ai", text: "Hello! I'm here to help you with career planning. How can I assist you today?" }]);
        }
        setError(null);
      } catch (err) {
        console.error("Failed to initialize session:", err);
        setError("Failed to connect to the server. Please make sure the backend is running.");
        // Add a fallback message
        setMessages([{ sender: "ai", text: "Hello! I'm here to help you with career planning. How can I assist you today?" }]);
      } finally {
        setIsLoading(false);
      }
    };

    initializeChat();
  }, []);

  const typeOutMessage = (text: string) => {
    const words = text.split(" ");
    let currentWordIndex = 0;

    // Add empty AI message first - this will be at the end of the messages array
    setMessages((prev) => [...prev, { sender: "ai", text: "" }]);

    // Type out each word at a fixed interval
    typingIntervalRef.current = setInterval(() => {
      if (currentWordIndex >= words.length) {
        if (typingIntervalRef.current) {
          clearInterval(typingIntervalRef.current);
          typingIntervalRef.current = null;
        }
        return;
      }

      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        const lastMessageIndex = updatedMessages.length - 1;

        // Update the last message (which should be the AI message we just added)
        if (lastMessageIndex >= 0 && updatedMessages[lastMessageIndex].sender === "ai") {
          updatedMessages[lastMessageIndex].text +=
            (updatedMessages[lastMessageIndex].text ? " " : "") + words[currentWordIndex];
        }

        return updatedMessages;
      });

      currentWordIndex++;
    }, 100);
  };

  const sendMessage = async () => {
    const trimmedInput = input.trim();
    if (trimmedInput === "" || isLoading) {
      return;
    }

    // Add user message to chat
    const userMessage: Message = { sender: "user", text: trimmedInput };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      let response;

      if (!sessionId) {
        // Start a new session if we don't have one
        response = await apiService.startSession(trimmedInput);
        setSessionId(response.session_id);
      } else {
        // Send response to existing session
        response = await apiService.sendResponse(sessionId, trimmedInput);
      }

      // Helper function to calculate typing duration based on word count
      const calculateTypingDuration = (text: string): number => {
        const wordCount = text.split(/\s+/).length;
        const typingTime = wordCount * 100; // 100ms per word
        return typingTime + 1000; // Add 1 second buffer
      };

      // Check if conversation is completed
      if (response.completed) {
        setIsCompleted(true);

        let delay = 0;

        // Show final report or completion message
        let completionMessage = "";
        if (response.final_report) {
          completionMessage = response.final_report;
        } else if (response.message) {
          completionMessage = response.message;
        } else {
          completionMessage = "Thank you! Your career planning session is complete.";
        }

        typeOutMessage(completionMessage);
        delay += calculateTypingDuration(completionMessage);

        // Show career predictions if available
        if (response.career_predictions && response.career_predictions.length > 0) {
          const predictionsText = response.career_predictions
            .map((pred) => `${pred.career_title} (Match: ${pred.match_score}%)`)
            .join(", ");
          const fullPredictionsText = `Career Predictions: ${predictionsText}`;

          setTimeout(() => {
            typeOutMessage(fullPredictionsText);
          }, delay);
          delay += calculateTypingDuration(fullPredictionsText);
        }

        // CRITICAL FIX: Show academic_message with proper timing to prevent overlap
        if (response.academic_message) {
          const academicMsg = response.academic_message; // Type narrowed to string
          setTimeout(() => {
            typeOutMessage(academicMsg);
          }, delay);
          delay += calculateTypingDuration(academicMsg);
        }

        // CRITICAL FIX: Show skill_message AFTER academic finishes
        if (response.skill_message) {
          const skillMsg = response.skill_message; // Type narrowed to string
          setTimeout(() => {
            typeOutMessage(skillMsg);
          }, delay);
          // No need to update delay after last message
        }
      } else if (response.question) {
        // Agent asked a follow-up question
        typeOutMessage(response.question);
      } else if (response.message) {
        // Generic message response
        typeOutMessage(response.message);
      } else {
        // Fallback response
        typeOutMessage("Thank you for your response. Let me process that...");
      }

    } catch (err) {
      console.error("Error sending message:", err);
      setError(err instanceof Error ? err.message : "Failed to send message");
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: "Sorry, I encountered an error. Please try again or check if the backend server is running."
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !isLoading) {
      sendMessage();
    }
  };

  useEffect(() => {
    return () => {
      // Cleanup if component unmounts
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
      // End session when component unmounts
      if (sessionId) {
        apiService.endSession(sessionId).catch(console.error);
      }
    };
  }, [sessionId]);

  return (
    <div className="mx-auto flex h-[70vh] w-full flex-col rounded-xl border border-neutral-400/20 bg-neutral-800 p-4 text-white">
      {/* Error message */}
      {error && (
        <div className="mb-2 rounded-md border border-red-500/50 bg-red-500/10 p-2 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Message list expands to fill available space */}
      <MessageList messages={messages} isLoading={isLoading} />

      {/* Input stays pinned at bottom */}
      <div className="mt-auto flex space-x-2 pt-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder={isLoading ? "Waiting for response..." : "Type your message..."}
          disabled={isLoading || isCompleted}
          className="font-fredoka w-full rounded-lg border border-neutral-400/20 bg-neutral-400/20 p-3 text-white placeholder:text-white disabled:opacity-50"
        />
        <button
          type="button"
          onClick={sendMessage}
          disabled={isLoading || isCompleted || !input.trim()}
          className="font-fredoka rounded-lg border border-neutral-400/20 bg-neutral-400/20 px-4 text-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? "..." : "Send"}
        </button>
      </div>

      {/* Show prediction button when completed or after 2+ user messages */}
      {(isCompleted || userMessagesCount >= 2) && (
        <button
          onClick={() => navigate("/prediction")}
          className="chat-button font-fredoka mt-6 w-full rounded-lg p-3 font-bold text-white"
        >
          Prediction is Ready
        </button>
      )}
    </div>
  );
};

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

const MessageList = ({ messages, isLoading }: MessageListProps) => {
  return (
    <ul ref={autoScrollListRef} className="mb-4 space-y-2 overflow-y-auto rounded-md pt-4">
      {messages.map((msg, index) => (
        <MessageItem key={`${index}-${msg.sender}-${msg.text}`} message={msg} />
      ))}
      {isLoading && (
        <li className="font-fredoka rounded-md border border-neutral-400/20 bg-neutral-400/10 p-2">
          <span className="animate-pulse">Agent is thinking...</span>
        </li>
      )}
    </ul>
  );
};

interface MessageItemProps {
  message: Message;
}

const MessageItem = ({ message }: MessageItemProps) => {
  return (
    <li
      className={`font-fredoka rounded-md p-2 break-words ${
        message.sender === "user"
          ? "font-fredoka self-end border border-sky-400/20 bg-sky-400/10"
          : "font-fredoka border border-neutral-400/20 bg-neutral-400/10"
      }`}
    >
      {message.text}
    </li>
  );
};

export default PreviewUseAutoScroll;
