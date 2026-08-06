export interface CitedPassage {
  chapter: number;
  verse_number: number;
  passage_type: "translation" | "commentary";
  text: string;
  similarity: number;
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: CitedPassage[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    let detail = "The reasoning engine couldn't answer that.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // response body wasn't JSON — keep the default message
    }
    throw new ApiError(detail, response.status);
  }

  return response.json();
}
