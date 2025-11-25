import type {
  SessionStartRequest,
  SessionInitializeRequest,
  UserResponse,
  InteractiveResponse,
  SessionStatus,
  ApiError,
} from "../types/chat";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `HTTP error! status: ${response.status}`,
        status: response.status,
      }));
      throw new Error(error.detail);
    }
    return response.json();
  }

  /**
   * Initialize a new agent session
   * The agent will start the conversation
   * @param language - User's preferred language ("en" or "si")
   */
  async initializeSession(
    language: "en" | "si" = "en",
  ): Promise<InteractiveResponse> {
    const request: SessionInitializeRequest = {
      language,
    };

    const response = await fetch(`${this.baseUrl}/session/initialize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    return this.handleResponse<InteractiveResponse>(response);
  }

  /**
   * Start a new interactive session with an initial message
   * @param initialMessage - The user's first message
   */
  async startSession(
    initialMessage: string,
  ): Promise<InteractiveResponse> {
    const request: SessionStartRequest = {
      initial_message: initialMessage,
      mode: "interactive",
    };

    const response = await fetch(`${this.baseUrl}/sessions/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    return this.handleResponse<InteractiveResponse>(response);
  }

  /**
   * Send a user response to an active session
   * @param sessionId - The session ID
   * @param userResponse - The user's response
   * @param language - User's preferred language ("en" or "si")
   */
  async sendResponse(
    sessionId: string,
    userResponse: string,
    language: "en" | "si" = "en",
  ): Promise<InteractiveResponse> {
    const request: UserResponse = {
      response: userResponse,
      language,
    };

    const response = await fetch(
      `${this.baseUrl}/sessions/${sessionId}/respond`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      },
    );
    return this.handleResponse<InteractiveResponse>(response);
  }

  /**
   * Update language preference for an active session
   * @param sessionId - The session ID
   * @param language - New language preference ("en" or "si")
   */
  async updateLanguage(
    sessionId: string,
    language: "en" | "si",
  ): Promise<{ success: boolean; language: string }> {
    const response = await fetch(
      `${this.baseUrl}/sessions/${sessionId}/language`,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ language }),
      },
    );
    return this.handleResponse<{ success: boolean; language: string }>(
      response,
    );
  }

  /**
   * Get the current status of a session
   * @param sessionId - The session ID
   */
  async getSessionStatus(sessionId: string): Promise<SessionStatus> {
    const response = await fetch(
      `${this.baseUrl}/sessions/${sessionId}/status`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      },
    );
    return this.handleResponse<SessionStatus>(response);
  }

  /**
   * End a session
   * @param sessionId - The session ID
   */
  async endSession(
    sessionId: string,
  ): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    });
    return this.handleResponse<{ success: boolean; message: string }>(
      response,
    );
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await fetch(`${this.baseUrl}/health`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
    return this.handleResponse<{ status: string; timestamp: string }>(
      response,
    );
  }
}

// Create a singleton instance
const apiService = new ApiService();

export default apiService;
