// API Types for Career Planning Chat System

export interface Message {
  sender: "user" | "ai";
  text: string;
}

export interface SessionStartRequest {
  initial_message: string;
  mode?: "interactive" | "batch";
}

export interface UserResponse {
  response: string;
}

export interface InteractiveResponse {
  success: boolean;
  session_id: string;
  question?: string;
  awaiting_user_response: boolean;
  conversation_state?: string;
  progress?: {
    current_question?: number;
    total_questions?: number;
    stage?: string;
  };
  final_report?: string;
  completed: boolean;
  career_predictions?: Array<{
    career_title: string;
    match_score: number;
    description: string;
  }>;
  career_title?: string;
  academic_plan?: Record<string, unknown>;
  skill_plan?: Record<string, unknown>;
  academic_message?: string;
  skill_message?: string;
  message?: string;
}

export interface SessionStatus {
  session_id: string;
  status: string;
  current_stage?: string;
  total_interactions?: number;
  created_at?: string;
}

export interface ChatSession {
  session_id: string;
  messages: Message[];
  isActive: boolean;
  isCompleted: boolean;
}

export interface ApiError {
  detail: string;
  status?: number;
}
