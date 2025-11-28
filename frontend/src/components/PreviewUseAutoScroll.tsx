import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { autoScrollListRef } from "../hooks/use-auto-scroll";
import { useNavigate } from "react-router-dom";

interface Message {
  sender: "user" | "ai";
  text: string;
}

const PreviewUseAutoScroll = () => {
  const [messages, setMessages] = useState<Message[]>([
    { sender: "ai", text: "Welcome to the chat!" },
    { sender: "ai", text: "Feel free to add new messages." },
  ]);
  const [input, setInput] = useState("");
  const navigate = useNavigate();
  const userMessagesCount = messages.filter((msg) => msg.sender === "user").length;

  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const sendMessage = () => {
    const trimmedInput = input.trim();
    if (trimmedInput === "") {
      return;
    }

    const userMessage: Message = { sender: "user", text: trimmedInput };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    // Simulate AI response
    const aiResponse = `Sint nisi eu cillum nulla officia incididunt irure laboris enim cillum cupidatat occaecat. 
Duis adipisicing veniam exercitation quis anim. Exercitation consectetur tempor et consectetur dolor. 
Cupidatat culpa eiusmod ex enim occaecat dolor sunt. Et et commodo qui ipsum nostrud ut et incididunt est cupidatat excepteur laborum. 
Anim ullamco aliqua ad sit sint cupidatat esse esse.`;

    // Break down the AI response into words
    const words = aiResponse.split(" ");
    let currentWordIndex = 0;

    // Add a placeholder AI message first, so we have something to update
    const newAiMessageIndex = messages.length + 1; // next index after user's message
    setMessages((prev) => [...prev, { sender: "ai", text: "" }]);

    // Type out each word at a fixed interval
    typingIntervalRef.current = setInterval(() => {
      setMessages((prevMessages) => {
        // Ensure the AI message exists
        if (!prevMessages[newAiMessageIndex]) {
          return prevMessages;
        }

        // Update the AI message with the next word
        const updatedMessages = [...prevMessages];
        const currentAiMessage = updatedMessages[newAiMessageIndex];

        currentAiMessage.text += (currentAiMessage.text ? " " : "") + words[currentWordIndex];

        currentWordIndex++;

        // If we've reached the end of all words, clear the interval
        if (currentWordIndex >= words.length) {
          if (typingIntervalRef.current) {
            clearInterval(typingIntervalRef.current);
            typingIntervalRef.current = null;
          }
        }

        return updatedMessages;
      });
    }, 100); // 100ms per word
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  useEffect(() => {
    return () => {
      // Cleanup if component unmounts
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
    };
  }, []);

  return (
    <div className="mx-auto flex h-[70vh] w-full flex-col rounded-xl border border-neutral-400/20 bg-neutral-800 p-4 text-white">
      {/* Message list expands to fill available space */}
      <MessageList messages={messages} />

      {/* Input stays pinned at bottom */}
      <div className="mt-auto flex space-x-2 pt-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Type your message..."
          className="font-space-grotesk w-full rounded-lg border border-neutral-400/20 bg-neutral-400/20 p-3 text-white placeholder:text-white"
        />
        <button
          type="button"
          onClick={sendMessage}
          className="font-space-grotesk rounded-lg border border-neutral-400/20 bg-neutral-400/20 px-4 text-white"
        >
          Send
        </button>
      </div>
      {userMessagesCount >= 2 && (
        <button
          onClick={() => navigate("/prediction")}
          className="chat-button font-space-grotesk mt-6 w-full rounded-lg p-3 font-bold text-white"
        >
          Prediction is Ready
        </button>
      )}
    </div>
  );
};

interface MessageListProps {
  messages: Message[];
}

const MessageList = ({ messages }: MessageListProps) => {
  return (
    <ul ref={autoScrollListRef} className="mb-4 space-y-2 overflow-y-auto rounded-md pt-4">
      {messages.map((msg, index) => (
        <MessageItem key={`${index}-${msg.sender}-${msg.text}`} message={msg} />
      ))}
    </ul>
  );
};

interface MessageItemProps {
  message: Message;
}

const MessageItem = ({ message }: MessageItemProps) => {
  return (
    <li
      className={`font-space-grotesk rounded-md p-2 break-words ${
        message.sender === "user"
          ? "font-space-grotesk self-end border border-teal-400/20 bg-teal-400/10"
          : "font-space-grotesk border border-neutral-400/20 bg-neutral-400/10"
      }`}
    >
      {message.text}
    </li>
  );
};

export default PreviewUseAutoScroll;
