const API_URL = 'http://localhost:8000';

export interface ApiResponse<T> {
    data: T;
    error?: string;
}

export async function fetchApi<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<ApiResponse<T>> {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return { data };
    } catch (error) {
        return {
            data: {} as T,
            error: error instanceof Error ? error.message : 'An unknown error occurred',
        };
    }
}

// Example API functions
export const api = {
    // Get welcome message
    getWelcome: () => fetchApi<{ message: string }>('/'),
    
    // Send chat message
    sendMessage: (message: string) => 
        fetchApi<{ reply: string }>('/chat', {
            method: 'POST',
            body: JSON.stringify({ message }),
        }),
}; 