import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { autoScrollListRef } from "../hooks/use-auto-scroll";
import apiService from "../services/api";
import type { Message } from "../types/chat";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import LanguageToggle from "./LanguageToggle";

// Helper function to remove "undefined" text from messages (final safety net)
const cleanUndefinedText = (text: string): string => {
  if (!text) return text;

  // Remove "undefined" (case-insensitive) with surrounding whitespace
  let cleaned = text.replace(/\s*undefined\s*/gi, ' ');

  // Clean up multiple spaces/tabs (preserve newlines for markdown formatting)
  cleaned = cleaned.replace(/[ \t]+/g, ' ').trim();

  return cleaned;
};

const PreviewUseAutoScroll = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCompleted, setIsCompleted] = useState(false);

  // Language state with localStorage persistence
  const [language, setLanguage] = useState<"en" | "si">(() => {
    return (localStorage.getItem("preferredLanguage") as "en" | "si") || "en";
  });

  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Initialize session when component mounts
  useEffect(() => {
    const initializeChat = async () => {
      try {
        setIsLoading(true);
        const response = await apiService.initializeSession(language);
        setSessionId(response.session_id);

        // Add the agent's initial question
        if (response.question) {
          setMessages([{ sender: "ai", text: cleanUndefinedText(response.question) }]);
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

  const typeOutMessage = (text: string, onComplete?: () => void) => {
    // Clean any "undefined" text as a final safety net
    const cleanedText = cleanUndefinedText(text);

    // Split by spaces while preserving newlines as separate tokens
    // This regex splits on spaces but keeps newlines intact
    const tokens: string[] = [];
    const lines = cleanedText.split('\n');

    lines.forEach((line, lineIndex) => {
      const words = line.split(' ').filter(word => word.length > 0);
      tokens.push(...words);
      // Add newline token after each line except the last
      if (lineIndex < lines.length - 1) {
        tokens.push('\n');
      }
    });

    let currentTokenIndex = 0;

    // Add empty AI message first - this will be at the end of the messages array
    setMessages((prev) => [...prev, { sender: "ai", text: "" }]);

    // Type out each token at a fixed interval
    typingIntervalRef.current = setInterval(() => {
      if (currentTokenIndex >= tokens.length) {
        if (typingIntervalRef.current) {
          clearInterval(typingIntervalRef.current);
          typingIntervalRef.current = null;
        }
        // Call the completion callback if provided
        if (onComplete) {
          onComplete();
        }
        return;
      }

      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        const lastMessageIndex = updatedMessages.length - 1;

        // Update the last message (which should be the AI message we just added)
        if (lastMessageIndex >= 0 && updatedMessages[lastMessageIndex].sender === "ai") {
          const token = tokens[currentTokenIndex];
          // Safety check: only add token if it exists and is not empty
          if (token !== undefined && token !== "") {
            // If token is a newline, add it directly; otherwise add with space
            if (token === '\n') {
              updatedMessages[lastMessageIndex].text += '\n';
            } else {
              updatedMessages[lastMessageIndex].text +=
                (updatedMessages[lastMessageIndex].text && !updatedMessages[lastMessageIndex].text.endsWith('\n') ? " " : "") + token;
            }
          }
        }

        return updatedMessages;
      });

      currentTokenIndex++;
    }, 100);
  };

  // Auto-resize textarea as user types
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    // Reset height to get accurate scrollHeight
    textarea.style.height = 'auto';

    // Calculate the height based on content
    const lineHeight = 24; // Approximate line height in pixels
    const maxLines = 8;
    const maxHeight = lineHeight * maxLines;

    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${newHeight}px`;
  }, [input]);

  // Handle language change
  const handleLanguageChange = (newLang: "en" | "si") => {
    setLanguage(newLang);
    localStorage.setItem("preferredLanguage", newLang);

    // Update backend session if exists
    if (sessionId) {
      apiService.updateLanguage(sessionId, newLang).catch((err) => {
        console.error("Failed to update language:", err);
      });
    }
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

    // Reset textarea height after sending
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setIsLoading(true);
    setError(null);

    try {
      let response;

      if (!sessionId) {
        // Start a new session if we don't have one
        response = await apiService.startSession(trimmedInput);
        setSessionId(response.session_id);
      } else {
        // Send response to existing session with language
        response = await apiService.sendResponse(sessionId, trimmedInput, language);
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

        // CRITICAL FIX: Show academic_message FIRST, then skill_message AFTER it completes
        // Using callback chaining to ensure proper sequential order and prevent truncation
        if (response.academic_message && response.skill_message) {
          const academicMsg = response.academic_message;
          const skillMsg = response.skill_message;

          // Start academic message after calculated delay
          setTimeout(() => {
            // Type academic message, and when it completes, start skill message
            typeOutMessage(academicMsg, () => {
              // Academic is complete, now start skill message
              setTimeout(() => {
                typeOutMessage(skillMsg);
              }, 1000); // 1 second delay between messages for better UX
            });
          }, delay);
        } else if (response.academic_message) {
          // Only academic message available
          const academicMsg = response.academic_message;
          setTimeout(() => {
            typeOutMessage(academicMsg);
          }, delay);
        } else if (response.skill_message) {
          // Only skill message available
          const skillMsg = response.skill_message;
          setTimeout(() => {
            typeOutMessage(skillMsg);
          }, delay);
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

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Shift+Enter: Add new line (default behavior)
    // Enter alone: Send message
    if (e.key === "Enter" && !e.shiftKey && !isLoading) {
      e.preventDefault(); // Prevent adding a new line
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
    <>
      {/* Language Toggle - positioned in page background */}
      <LanguageToggle
        currentLanguage={language}
        onLanguageChange={handleLanguageChange}
      />

      {/* Chat Interface */}
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
        <div className="mt-auto flex items-end space-x-2 pt-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={isLoading ? "Waiting for response..." : "Type your message..."}
            disabled={isLoading || isCompleted}
            rows={1}
            className="font-fredoka w-full rounded-lg border border-neutral-400/20 bg-neutral-400/20 p-3 text-white placeholder:text-white disabled:opacity-50 resize-none overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-neutral-700/50 [&::-webkit-scrollbar-track]:rounded-full [&::-webkit-scrollbar-thumb]:bg-neutral-500/50 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:hover:bg-neutral-400/70"
            style={{ minHeight: '48px', maxHeight: '192px' }}
          />
          <button
            type="button"
            onClick={sendMessage}
            disabled={isLoading || isCompleted || !input.trim()}
            className="font-fredoka rounded-lg border border-neutral-400/20 bg-neutral-400/20 px-4 py-3 text-white disabled:opacity-50 disabled:cursor-not-allowed h-[48px]"
          >
            {isLoading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </>
  );
};

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

const MessageList = ({ messages, isLoading }: MessageListProps) => {
  return (
    <ul ref={autoScrollListRef} className="mb-4 space-y-2 overflow-y-auto rounded-md pt-4 [&::-webkit-scrollbar]:hidden [scrollbar-width:none] [-ms-overflow-style:none]">
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
      <div className="markdown-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkBreaks]}
          components={{
            // Style headings
            h1: ({node, ...props}) => <h1 className="text-2xl font-bold mb-3 mt-4 text-sky-400" {...props} />,
            h2: ({node, ...props}) => <h2 className="text-xl font-bold mb-2 mt-3 text-sky-300" {...props} />,
            h3: ({node, ...props}) => <h3 className="text-lg font-semibold mb-2 mt-2 text-sky-200" {...props} />,
            // Style lists
            ul: ({node, ...props}) => <ul className="list-disc list-inside mb-2 ml-2 space-y-1" {...props} />,
            ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-2 ml-2 space-y-1" {...props} />,
            li: ({node, ...props}) => <li className="ml-2" {...props} />,
            // Style paragraphs
            p: ({node, ...props}) => <p className="mb-2 leading-relaxed" {...props} />,
            // Style strong/bold text
            strong: ({node, ...props}) => <strong className="font-bold text-white" {...props} />,
            // Style emphasis/italic text
            em: ({node, ...props}) => <em className="italic text-neutral-200" {...props} />,
            // Style code blocks
            code: ({node, className, children, ...props}) => {
              const isInline = !className;
              return isInline ? (
                <code className="bg-neutral-700 px-1 py-0.5 rounded text-sky-300" {...props}>
                  {children}
                </code>
              ) : (
                <code className="block bg-neutral-700 p-2 rounded my-2 text-sky-300 overflow-x-auto" {...props}>
                  {children}
                </code>
              );
            },
            // Style links
            a: ({node, ...props}) => <a className="text-sky-400 hover:text-sky-300 underline" {...props} />,
            // Style blockquotes
            blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-sky-400 pl-3 italic my-2" {...props} />,
            // Style horizontal rules
            hr: ({node, ...props}) => <hr className="my-4 border-neutral-600" {...props} />,
          }}
        >
          {message.text}
        </ReactMarkdown>
      </div>
    </li>
  );
};

export default PreviewUseAutoScroll;
