import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { Send, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { useLocation } from "react-router-dom";
import { useLanguage } from "../../context/LanguageContext";

interface Message {
    sender: "user" | "ai";
    text: string;
}

const ChatPage = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const { language, setLanguage } = useLanguage();
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const location = useLocation();
    const hasInitialized = useRef(false);
    const sessionIdRef = useRef<string | null>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isTyping]);

    // Initialize session on mount
    useEffect(() => {
        if (!hasInitialized.current) {
            hasInitialized.current = true;

            // If we have an initial message from landing page, use standard chat flow
            if (location.state?.initialMessage) {
                setInput(location.state.initialMessage);
                sendMessage(location.state.initialMessage);
            } else {
                // Otherwise, start agent-initiated session
                startAgentSession();
            }
        }
    }, [location.state]);

    const startAgentSession = async () => {
        setIsTyping(true);
        try {
            const response = await fetch("http://localhost:8000/session/initialize", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    language: language,
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to initialize session");
            }

            const data = await response.json();
            sessionIdRef.current = data.session_id;

            if (data.question) {
                setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
            }
        } catch (error) {
            console.error("Error initializing session:", error);
            setMessages((prev) => [
                ...prev,
                { sender: "ai", text: "Sorry, I couldn't start the session. Please try refreshing." },
            ]);
        } finally {
            setIsTyping(false);
        }
    };

    const sendMessage = async (messageText: string = input) => {
        const trimmedInput = messageText.trim();
        if (trimmedInput === "") {
            return;
        }

        const userMessage: Message = { sender: "user", text: trimmedInput };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsTyping(true);

        try {
            // Determine endpoint based on whether we have a session ID (interactive mode) or not
            const url = sessionIdRef.current
                ? `http://localhost:8000/sessions/${sessionIdRef.current}/respond`
                : "http://localhost:8000/chat";

            const body = sessionIdRef.current
                ? { response: trimmedInput, language: language }
                : { message: trimmedInput, language: language };

            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(body),
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            // Handle streaming response (for /chat endpoint)
            if (!sessionIdRef.current) {
                const reader = response.body?.getReader();
                if (!reader) throw new Error("Response body is null");

                const decoder = new TextDecoder();
                let aiResponseText = "";
                let isFirstChunk = true;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split("\n\n");

                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            const dataStr = line.replace("data: ", "");
                            if (dataStr === "[DONE]") break;

                            try {
                                const data = JSON.parse(dataStr);

                                // Handle session ID from start event
                                if (data.type === "start" && data.session_id) {
                                    sessionIdRef.current = data.session_id;
                                }
                                // Handle waiting for user (question)
                                else if (data.type === "waiting_for_user" && data.question) {
                                    // If we were building a text response, ensure it's finished
                                    // If the question is different from what we've shown, add it
                                    if (data.question !== aiResponseText) {
                                        setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
                                    }
                                }
                                // Handle standard text content (streaming)
                                else if (data.type === "text" && data.content) {
                                    aiResponseText += data.content;

                                    setMessages((prev) => {
                                        const newMessages = [...prev];
                                        // Check if the last message is from AI and we are streaming
                                        const lastMessage = newMessages[newMessages.length - 1];

                                        if (isFirstChunk || lastMessage.sender !== "ai") {
                                            newMessages.push({ sender: "ai", text: aiResponseText });
                                            isFirstChunk = false;
                                        } else {
                                            lastMessage.text = aiResponseText;
                                        }
                                        return newMessages;
                                    });
                                }
                                // Ignore 'log', 'progress', 'warning' types as requested
                            } catch (e) {
                                console.error("Error parsing JSON:", e);
                            }
                        }
                    }
                }
            } else {
                // Handle standard JSON response (for /respond endpoint)
                const data = await response.json();
                if (data.question) {
                    setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
                } else if (data.message) {
                    setMessages((prev) => [...prev, { sender: "ai", text: data.message }]);
                }
            }
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) => [
                ...prev,
                { sender: "ai", text: "Sorry, something went wrong. Please try again." },
            ]);
        } finally {
            setIsTyping(false);
        }
    };

    const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    };

    return (
        <div className="flex h-[calc(100vh-72px)] w-full flex-col items-center bg-gradient-to-br from-teal-50 via-cyan-50 to-teal-100 px-4 pb-6 pt-6">
            <div className="flex w-full max-w-4xl flex-1 flex-col overflow-hidden rounded-3xl bg-white/80 shadow-2xl backdrop-blur-xl ring-1 ring-white/50">

                {/* Header */}
                <div className="flex items-center justify-between border-b border-gray-100 bg-white/50 px-6 py-4 backdrop-blur-md">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-tr from-teal-400 to-cyan-500 shadow-lg shadow-teal-500/20">
                            <Sparkles className="h-5 w-5 text-white" />
                        </div>
                        <div>
                            <h2 className="font-space-grotesk text-2xl font-bold text-gray-800">Horizon</h2>
                            <p className="text-base font-medium text-teal-600">Discover your true potential</p>
                        </div>
                    </div>

                    {/* Language Toggle */}
                    <button
                        onClick={() => setLanguage(language === "en" ? "si" : "en")}
                        className="relative inline-flex cursor-pointer rounded-full bg-gray-100/80 p-1 backdrop-blur-sm ring-1 ring-gray-200 transition-all hover:ring-gray-300 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                    >
                        <span
                            className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${language === "en" ? "text-white" : "text-gray-500"
                                }`}
                        >
                            English
                            {language === "en" && (
                                <motion.div
                                    layoutId="active-language-chat"
                                    className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md"
                                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                                />
                            )}
                        </span>
                        <span
                            className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${language === "si" ? "text-white" : "text-gray-500"
                                }`}
                        >
                            සිංහල
                            {language === "si" && (
                                <motion.div
                                    layoutId="active-language-chat"
                                    className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md"
                                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                                />
                            )}
                        </span>
                    </button>
                </div>

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-6 scrollbar-hide">
                    <div className="space-y-6">
                        {messages.map((msg, index) => (
                            <div
                                key={index}
                                className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
                            >
                                <div
                                    className={`max-w-[80%] rounded-2xl px-6 py-4 text-lg select-text leading-relaxed shadow-sm ${msg.sender === "user"
                                        ? "bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-br-none"
                                        : "bg-white text-gray-700 ring-1 ring-gray-100 rounded-bl-none"
                                        }`}
                                >
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        {isTyping && (
                            <div className="flex justify-start">
                                <div className="max-w-[80%] rounded-2xl rounded-bl-none bg-white px-6 py-4 text-gray-500 shadow-sm ring-1 ring-gray-100">
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ repeat: Infinity, duration: 1.5 }}
                                    >
                                        Thinking...
                                    </motion.div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* Input Area */}
                <div className="border-t border-gray-100 bg-white/50 p-4 backdrop-blur-md">
                    <div className="relative mx-auto max-w-3xl">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyPress}
                            placeholder="Type your message..."
                            disabled={isTyping}
                            className="w-full rounded-full border-2 border-teal-600 bg-white py-4 pl-6 pr-14 text-gray-700 shadow-lg shadow-gray-200/50 ring-1 ring-gray-100 transition-all placeholder:text-gray-400 focus:outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-500/20 disabled:opacity-50"
                        />
                        <button
                            onClick={() => sendMessage()}
                            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 p-2.5 text-white shadow-md transition-all hover:scale-105 hover:shadow-lg active:scale-95 disabled:opacity-50"
                            disabled={!input.trim() || isTyping}
                        >
                            <Send className="h-5 w-5" />
                        </button>
                    </div>
                    <p className="mt-3 text-center text-xs text-gray-400">
                        AI can make mistakes. Consider checking important information.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ChatPage;
