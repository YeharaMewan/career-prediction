// import { useState, useRef, useEffect, useCallback, type KeyboardEvent } from "react";
// import { Send, Sparkles } from "lucide-react";
// import { motion } from "framer-motion";
// import { useLocation, useNavigate } from "react-router-dom";
// import { useLanguage } from "../../context/LanguageContext";
// import type { CareerPrediction } from "../../types/career";

// interface Message {
//     sender: "user" | "ai";
//     text: string;
// }

// const ChatPage = () => {
//     const [messages, setMessages] = useState<Message[]>([]);
//     const [input, setInput] = useState("");
//     const [isTyping, setIsTyping] = useState(false);
//     const [debouncedIsTyping, setDebouncedIsTyping] = useState(false);
//     const [careerPredictions, setCareerPredictions] = useState<CareerPrediction[]>([]);
//     const [showPredictionsButton, setShowPredictionsButton] = useState(false);
//     const [questionCount, setQuestionCount] = useState(0);
//     const [isPredictionsLoading, setIsPredictionsLoading] = useState(false);
//     const { language, setLanguage } = useLanguage();
//     const messagesEndRef = useRef<HTMLDivElement>(null);
//     const location = useLocation();
//     const navigate = useNavigate();
//     const hasInitialized = useRef(false);
//     const sessionIdRef = useRef<string | null>(null);
//     const textareaRef = useRef<HTMLTextAreaElement>(null);
//     const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

//     const scrollToBottom = () => {
//         messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//     };

//     useEffect(() => {
//         scrollToBottom();
//     }, [messages, debouncedIsTyping]);

//     // Auto-resize textarea
//     useEffect(() => {
//         if (textareaRef.current) {
//             textareaRef.current.style.height = "auto";
//             textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
//         }
//     }, [input]);

//     // Debounce isTyping to prevent flicker
//     useEffect(() => {
//         if (isTyping) {
//             // Immediately show typing indicator
//             setDebouncedIsTyping(true);
//         } else {
//             // Delay hiding to ensure minimum display duration (300ms)
//             debounceTimeoutRef.current = setTimeout(() => {
//                 setDebouncedIsTyping(false);
//             }, 300);
//         }

//         return () => {
//             if (debounceTimeoutRef.current) {
//                 clearTimeout(debounceTimeoutRef.current);
//                 debounceTimeoutRef.current = null;
//             }
//         };
//     }, [isTyping]);

//     // Immediately hide thinking indicator when AI message arrives
//     useEffect(() => {
//         if (messages.length > 0) {
//             const lastMessage = messages[messages.length - 1];
//             // If the last message is from AI and we're showing typing indicator, hide it immediately
//             if (lastMessage.sender === "ai" && debouncedIsTyping) {
//                 // Clear any pending timeout
//                 if (debounceTimeoutRef.current) {
//                     clearTimeout(debounceTimeoutRef.current);
//                     debounceTimeoutRef.current = null;
//                 }
//                 // Immediately hide
//                 setDebouncedIsTyping(false);
//             }
//         }
//     }, [messages, debouncedIsTyping]);


//     const startAgentSession = useCallback(async () => {
//         setIsTyping(true);
//         try {
//             const response = await fetch("http://localhost:8000/session/initialize", {
//                 method: "POST",
//                 headers: {
//                     "Content-Type": "application/json",
//                 },
//                 body: JSON.stringify({
//                     language: language,
//                 }),
//             });

//             if (!response.ok) {
//                 throw new Error("Failed to initialize session");
//             }

//             const data = await response.json();
//             sessionIdRef.current = data.session_id;

//             if (data.question) {
//                 setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
//             }
//         } catch (error) {
//             console.error("Error initializing session:", error);
//             setMessages((prev) => [
//                 ...prev,
//                 { sender: "ai", text: "Sorry, I couldn't start the session. Please try refreshing." },
//             ]);
//         } finally {
//             setIsTyping(false);
//         }
//     }, [language]);

//     const sendMessage = useCallback(async (messageText: string = input) => {
//         const trimmedInput = messageText.trim();
//         if (trimmedInput === "") {
//             return;
//         }

//         const userMessage: Message = { sender: "user", text: trimmedInput };
//         setMessages((prev) => [...prev, userMessage]);
//         setInput("");
//         setIsTyping(true);

//         try {
//             // Determine endpoint based on whether we have a session ID (interactive mode) or not
//             const url = sessionIdRef.current
//                 ? `http://localhost:8000/sessions/${sessionIdRef.current}/respond`
//                 : "http://localhost:8000/chat";

//             const body = sessionIdRef.current
//                 ? { response: trimmedInput, language: language }
//                 : { message: trimmedInput, language: language };

//             const response = await fetch(url, {
//                 method: "POST",
//                 headers: {
//                     "Content-Type": "application/json",
//                 },
//                 body: JSON.stringify(body),
//             });

//             if (!response.ok) {
//                 throw new Error("Network response was not ok");
//             }

//             // Handle streaming response (for /chat endpoint)
//             if (!sessionIdRef.current) {
//                 const reader = response.body?.getReader();
//                 if (!reader) throw new Error("Response body is null");

//                 const decoder = new TextDecoder();
//                 let aiResponseText = "";
//                 let isFirstChunk = true;

//                 while (true) {
//                     const { done, value } = await reader.read();
//                     if (done) break;

//                     const chunk = decoder.decode(value, { stream: true });
//                     const lines = chunk.split("\n\n");

//                     for (const line of lines) {
//                         if (line.startsWith("data: ")) {
//                             const dataStr = line.replace("data: ", "");
//                             if (dataStr === "[DONE]") break;

//                             try {
//                                 const data = JSON.parse(dataStr);

//                                 // Handle session ID from start event
//                                 if (data.type === "start" && data.session_id) {
//                                     sessionIdRef.current = data.session_id;
//                                 }
//                                 // Handle waiting for user (question)
//                                 else if (data.type === "waiting_for_user" && data.question) {
//                                     // If we were building a text response, ensure it's finished
//                                     // If the question is different from what we've shown, add it
//                                     if (data.question !== aiResponseText) {
//                                         setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
//                                     }
//                                 }
//                                 // Handle standard text content (streaming)
//                                 else if (data.type === "text" && data.content) {
//                                     aiResponseText += data.content;

//                                     setMessages((prev) => {
//                                         const newMessages = [...prev];
//                                         // Check if the last message is from AI and we are streaming
//                                         const lastMessage = newMessages[newMessages.length - 1];

//                                         if (isFirstChunk || lastMessage.sender !== "ai") {
//                                             newMessages.push({ sender: "ai", text: aiResponseText });
//                                             isFirstChunk = false;
//                                         } else {
//                                             lastMessage.text = aiResponseText;
//                                         }
//                                         return newMessages;
//                                     });
//                                 }
//                                 // Ignore 'log', 'progress', 'warning' types as requested
//                             } catch (e) {
//                                 console.error("Error parsing JSON:", e);
//                             }
//                         }
//                     }
//                 }
//             } else {
//                 // Handle standard JSON response (for /respond endpoint)
//                 const data = await response.json();

//                 // Check if response contains career predictions
//                 const hasPredictions = data.career_predictions &&
//                     Array.isArray(data.career_predictions) &&
//                     data.career_predictions.length > 0;

//                 if (hasPredictions) {
//                     // Career predictions received - show brief message ONLY
//                     setCareerPredictions(data.career_predictions);
//                     setShowPredictionsButton(true);
//                     setIsPredictionsLoading(false);

//                     setMessages((prev) => [...prev, {
//                         sender: "ai",
//                         text: "✨ Your career matches are ready! Click the button below to explore your personalized recommendations."
//                     }]);

//                     // IMPORTANT: Do NOT process data.question or data.message
//                     // The question field contains full career details which we're suppressing

//                 } else if (data.question) {
//                     // Normal question flow (only when NO predictions)
//                     setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
//                     setQuestionCount(prev => prev + 1);

//                     // Show loading when question 12 is being answered
//                     if (questionCount === 11) {  // After answering question 12
//                         setIsPredictionsLoading(true);
//                     }
//                 } else if (data.message) {
//                     setMessages((prev) => [...prev, { sender: "ai", text: data.message }]);
//                 }
//             }
//         } catch (error) {
//             console.error("Error sending message:", error);
//             setMessages((prev) => [
//                 ...prev,
//                 { sender: "ai", text: "Sorry, something went wrong. Please try again." },
//             ]);
//         } finally {
//             setIsTyping(false);
//         }
//     }, [input, language]);

//     // Initialize session on mount
//     useEffect(() => {
//         if (!hasInitialized.current) {
//             hasInitialized.current = true;

//             // If we have an initial message from landing page, use standard chat flow
//             if (location.state?.initialMessage) {
//                 setInput(location.state.initialMessage);
//                 sendMessage(location.state.initialMessage);
//             } else {
//                 // Otherwise, start agent-initiated session
//                 startAgentSession();
//             }
//         }
//     }, [location.state, sendMessage, startAgentSession]);

//     const handleViewPredictions = () => {
//         navigate("/prediction", {
//             state: {
//                 predictions: careerPredictions,
//                 sessionId: sessionIdRef.current,
//                 language: language
//             }
//         });
//     };

//     const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
//         if (e.key === "Enter" && !e.shiftKey) {
//             e.preventDefault();
//             sendMessage();
//         }
//     };

//     return (
//         <div className="flex h-[calc(100vh-72px)] w-full flex-col items-center bg-gradient-to-br from-teal-50 via-cyan-50 to-teal-100 px-4 pb-6 pt-6">
//             <div className="flex w-full max-w-4xl flex-1 flex-col overflow-hidden rounded-3xl bg-white/80 shadow-2xl backdrop-blur-xl ring-1 ring-white/50">

//                 {/* Header */}
//                 <div className="flex items-center justify-between border-b border-gray-100 bg-white/50 px-6 py-4 backdrop-blur-md">
//                     <div className="flex items-center gap-3">
//                         <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-tr from-teal-400 to-cyan-500 shadow-lg shadow-teal-500/20">
//                             <Sparkles className="h-5 w-5 text-white" />
//                         </div>
//                         <div>
//                             <h2 className="font-space-grotesk text-2xl font-bold text-gray-800">Horizon</h2>
//                             <p className="text-base font-medium text-teal-600">Discover your true potential</p>
//                         </div>
//                     </div>

//                     {/* Language Toggle */}
//                     <button
//                         onClick={() => setLanguage(language === "en" ? "si" : "en")}
//                         className="relative inline-flex cursor-pointer rounded-full bg-gray-100/80 p-1 backdrop-blur-sm ring-1 ring-gray-200 transition-all hover:ring-gray-300 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
//                     >
//                         <span
//                             className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${language === "en" ? "text-white" : "text-gray-500"
//                                 }`}
//                         >
//                             English
//                             {language === "en" && (
//                                 <motion.div
//                                     layoutId="active-language-chat"
//                                     className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md"
//                                     transition={{ type: "spring", stiffness: 300, damping: 30 }}
//                                 />
//                             )}
//                         </span>
//                         <span
//                             className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${language === "si" ? "text-white" : "text-gray-500"
//                                 }`}
//                         >
//                             සිංහල
//                             {language === "si" && (
//                                 <motion.div
//                                     layoutId="active-language-chat"
//                                     className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md"
//                                     transition={{ type: "spring", stiffness: 300, damping: 30 }}
//                                 />
//                             )}
//                         </span>
//                     </button>
//                 </div>

//                 {/* Messages Area */}
//                 <div className="flex-1 overflow-y-auto p-6 scrollbar-hide">
//                     <div className="space-y-6">
//                         {messages.map((msg, index) => (
//                             <div
//                                 key={index}
//                                 className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
//                             >
//                                 <div
//                                     className={`max-w-[80%] rounded-2xl px-6 py-4 text-lg select-text leading-relaxed shadow-sm ${msg.sender === "user"
//                                         ? "bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-br-none"
//                                         : "bg-white text-gray-700 ring-1 ring-gray-100 rounded-bl-none"
//                                         }`}
//                                 >
//                                     <div className="whitespace-pre-wrap">{msg.text}</div>
//                                 </div>
//                             </div>
//                         ))}
//                         {debouncedIsTyping && (
//                             <div className="flex justify-start">
//                                 <div className="max-w-[80%] rounded-2xl rounded-bl-none bg-white px-6 py-4 text-gray-500 shadow-sm ring-1 ring-gray-100">
//                                     <span className="bg-[length:200%_100%] bg-gradient-to-r from-teal-500 via-cyan-500 to-teal-500 bg-clip-text text-transparent font-medium animate-gradient-x">
//                                         {isPredictionsLoading ? "Analyzing your profile..." : "Thinking..."}
//                                     </span>
//                                 </div>
//                             </div>
//                         )}
//                         <div ref={messagesEndRef} />
//                     </div>
//                 </div>

//                 {/* View Career Options Button */}
//                 {showPredictionsButton && (
//                     <div className="flex justify-center border-t border-gray-100 bg-white/50 py-4 backdrop-blur-md">
//                         <button
//                             onClick={handleViewPredictions}
//                             className="bg-gradient-to-r from-teal-600 to-cyan-700 text-white px-8 py-3 rounded-full font-medium hover:scale-105 transition duration-300 shadow-lg hover:shadow-xl active:scale-95 flex items-center gap-2"
//                         >
//                             <span className="text-xl"></span>
//                             <span>View Career Options</span>
//                         </button>
//                     </div>
//                 )}

//                 {/* Input Area */}
//                 <div className="border-t border-gray-100 bg-white/50 p-4 backdrop-blur-md">
//                     <div className="relative mx-auto max-w-3xl">
//                         <textarea
//                             ref={textareaRef}
//                             value={input}
//                             onChange={(e) => setInput(e.target.value)}
//                             onKeyDown={handleKeyPress}
//                             placeholder="Type your message..."
//                             disabled={isTyping}
//                             rows={1}
//                             className="w-full resize-none rounded-3xl border-2 border-teal-600 bg-white py-4 pl-6 pr-14 text-gray-700 shadow-lg shadow-gray-200/50 ring-1 ring-gray-100 transition-all placeholder:text-gray-400 focus:outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-500/20 disabled:opacity-50 scrollbar-hide"
//                             style={{ minHeight: "60px", maxHeight: "120px" }}
//                         />
//                         <button
//                             onClick={() => sendMessage()}
//                             className="absolute right-2 top-2.5 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 p-2.5 text-white shadow-md transition-all hover:scale-105 hover:shadow-lg active:scale-95 disabled:opacity-50"
//                             disabled={!input.trim() || isTyping}
//                         >
//                             <Send className="h-5 w-5" />
//                         </button>
//                     </div>
//                     <p className="mt-3 text-center text-xs text-gray-400">
//                         AI can make mistakes. Consider checking important information.
//                     </p>
//                 </div>
//             </div>
//         </div >
//     );
// };

// export default ChatPage;

import { useState, useRef, useEffect, useCallback, type KeyboardEvent } from "react";
import { Send, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";
import { useLanguage } from "../../context/LanguageContext";
import type { CareerPrediction } from "../../types/career";

declare global {
  interface Window {
    lottie: any;
  }
}

interface Message {
    sender: "user" | "ai";
    text: string;
}

const ChatPage = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [debouncedIsTyping, setDebouncedIsTyping] = useState(false);
    const [careerPredictions, setCareerPredictions] = useState<CareerPrediction[]>([]);
    const [showPredictionsButton, setShowPredictionsButton] = useState(false);
    const [questionCount, setQuestionCount] = useState(0);
    const [isPredictionsLoading, setIsPredictionsLoading] = useState(false);
    const { language, setLanguage } = useLanguage();
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const location = useLocation();
    const navigate = useNavigate();
    const hasInitialized = useRef(false);
    const sessionIdRef = useRef<string | null>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const lottieRef = useRef<any>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, debouncedIsTyping]);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
        }
    }, [input]);

    // Load background animation
    useEffect(() => {
        if (lottieRef.current && window.lottie) {
            try {
                window.lottie.loadAnimation({
                    container: lottieRef.current,
                    renderer: "svg",
                    loop: true,
                    autoplay: true,
                    path: "/chatbot.json",
                });
            } catch (error) {
                console.error("Error loading animation:", error);
            }
        }
    }, []);

    // Debounce isTyping to prevent flicker
    useEffect(() => {
        if (isTyping) {
            // Immediately show typing indicator
            setDebouncedIsTyping(true);
        } else {
            // Delay hiding to ensure minimum display duration (300ms)
            debounceTimeoutRef.current = setTimeout(() => {
                setDebouncedIsTyping(false);
            }, 300);
        }

        return () => {
            if (debounceTimeoutRef.current) {
                clearTimeout(debounceTimeoutRef.current);
                debounceTimeoutRef.current = null;
            }
        };
    }, [isTyping]);

    // Immediately hide thinking indicator when AI message arrives
    useEffect(() => {
        if (messages.length > 0) {
            const lastMessage = messages[messages.length - 1];
            // If the last message is from AI and we're showing typing indicator, hide it immediately
            if (lastMessage.sender === "ai" && debouncedIsTyping) {
                // Clear any pending timeout
                if (debounceTimeoutRef.current) {
                    clearTimeout(debounceTimeoutRef.current);
                    debounceTimeoutRef.current = null;
                }
                // Immediately hide
                setDebouncedIsTyping(false);
            }
        }
    }, [messages, debouncedIsTyping]);


    const startAgentSession = useCallback(async () => {
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
    }, [language]);

    const sendMessage = useCallback(async (messageText: string = input) => {
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

                // Check if response contains career predictions
                const hasPredictions = data.career_predictions &&
                    Array.isArray(data.career_predictions) &&
                    data.career_predictions.length > 0;

                if (hasPredictions) {
                    // Career predictions received - show brief message ONLY
                    setCareerPredictions(data.career_predictions);
                    setShowPredictionsButton(true);
                    setIsPredictionsLoading(false);

                    setMessages((prev) => [...prev, {
                        sender: "ai",
                        text: "✨ Your career matches are ready! Click the button below to explore your personalized recommendations."
                    }]);

                    // IMPORTANT: Do NOT process data.question or data.message
                    // The question field contains full career details which we're suppressing

                } else if (data.question) {
                    // Normal question flow (only when NO predictions)
                    setMessages((prev) => [...prev, { sender: "ai", text: data.question }]);
                    setQuestionCount(prev => prev + 1);

                    // Show loading when question 12 is being answered
                    if (questionCount === 11) {  // After answering question 12
                        setIsPredictionsLoading(true);
                    }
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
    }, [input, language]);

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
    }, [location.state, sendMessage, startAgentSession]);

    const handleViewPredictions = () => {
        navigate("/prediction", {
            state: {
                predictions: careerPredictions,
                sessionId: sessionIdRef.current,
                language: language
            }
        });
    };

    const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="relative flex h-[calc(100vh-72px)] w-full flex-col items-center bg-gradient-to-br from-teal-50 via-cyan-50 to-teal-100 px-4 pb-6 pt-6 overflow-hidden">
            {/* Background Animation */}
            <div
                ref={lottieRef}
                className="absolute -left-2 bottom-0 opacity-20 pointer-events-none"
                style={{
                    width: "400px",
                    height: "400px",
                }}
            />

            {/* Content */}
            <div className="flex w-full max-w-4xl flex-1 flex-col overflow-hidden rounded-3xl bg-white/80 shadow-2xl backdrop-blur-xl ring-1 ring-white/50 relative z-10">

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
                                    <div className="whitespace-pre-wrap">{msg.text}</div>
                                </div>
                            </div>
                        ))}
                        {debouncedIsTyping && (
                            <div className="flex justify-start">
                                <div className="max-w-[80%] rounded-2xl rounded-bl-none bg-white px-6 py-4 text-gray-500 shadow-sm ring-1 ring-gray-100">
                                    <span className="bg-[length:200%_100%] bg-gradient-to-r from-teal-500 via-cyan-500 to-teal-500 bg-clip-text text-transparent font-medium animate-gradient-x">
                                        {isPredictionsLoading ? "Analyzing your profile..." : "Thinking..."}
                                    </span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* View Career Options Button */}
                {showPredictionsButton && (
                    <div className="flex justify-center border-t border-gray-100 bg-white/50 py-4 backdrop-blur-md">
                        <button
                            onClick={handleViewPredictions}
                            className="bg-gradient-to-r from-teal-600 to-cyan-700 text-white px-8 py-3 rounded-full font-medium hover:scale-105 transition duration-300 shadow-lg hover:shadow-xl active:scale-95 flex items-center gap-2"
                        >
                            <span className="text-xl"></span>
                            <span>View Career Options</span>
                        </button>
                    </div>
                )}

                {/* Input Area */}
                <div className="border-t border-gray-100 bg-white/50 p-4 backdrop-blur-md">
                    <div className="relative mx-auto max-w-3xl">
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyPress}
                            placeholder="Type your message..."
                            disabled={isTyping}
                            rows={1}
                            className="w-full resize-none rounded-3xl border-2 border-teal-600 bg-white py-4 pl-6 pr-14 text-gray-700 shadow-lg shadow-gray-200/50 ring-1 ring-gray-100 transition-all placeholder:text-gray-400 focus:outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-500/20 disabled:opacity-50 scrollbar-hide"
                            style={{ minHeight: "60px", maxHeight: "120px" }}
                        />
                        <button
                            onClick={() => sendMessage()}
                            className="absolute right-2 top-2.5 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 p-2.5 text-white shadow-md transition-all hover:scale-105 hover:shadow-lg active:scale-95 disabled:opacity-50"
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
        </div >
    );
};

export default ChatPage;