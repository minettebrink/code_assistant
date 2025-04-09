<script lang="ts">
    import { api } from '$lib/api';
    import { tick } from 'svelte';

    let messages: { text: string; isUser: boolean }[] = [];
    let newMessage = '';
    let loading = false;
    let messagesContainer: HTMLDivElement;

    async function scrollToBottom() {
        await tick();
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    async function handleSubmit() {
        if (!newMessage.trim()) return;

        messages = [...messages, { text: newMessage, isUser: true }];
        scrollToBottom();
        const userMessage = newMessage;
        newMessage = '';
        loading = true;

        try {
            // TODO: Replace with actual API call when backend endpoint is ready
            // const response = await api.sendMessage(userMessage);
            // messages = [...messages, { text: response.data.reply, isUser: false }];
            
            // Temporary mock response
            setTimeout(() => {
                messages = [...messages, { 
                    text: `Echo: ${userMessage}`, 
                    isUser: false 
                }];
                loading = false;
                scrollToBottom();
            }, 1000);
        } catch (error) {
            console.error('Error sending message:', error);
            loading = false;
        }
    }
</script>

<h1 class="chat-title">Code Assistant</h1>

<div class="chat-container">
    <div class="messages" bind:this={messagesContainer}>
        {#each messages as message}
            <div class="message {message.isUser ? 'user' : 'bot'}">
                <div class="message-content">
                    {message.text}
                </div>
            </div>
        {/each}
        {#if loading}
            <div class="message bot">
                <div class="message-content loading">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        {/if}
    </div>

    <form on:submit|preventDefault={handleSubmit} class="input-form">
        <input
            type="text"
            bind:value={newMessage}
            placeholder="Type your message..."
            disabled={loading}
        />
        <button type="submit" disabled={loading || !newMessage.trim()}>
            Send
        </button>
    </form>
</div>

<style>
    .chat-title {
        text-align: center;
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #333;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        height: 80vh; /* Increased height to 80% of the viewport height */
        /* Or use a fixed pixel value like: height: 750px; */
        max-width: 960px;
        margin: 2rem auto; /* Keeps vertical/horizontal centering */
        padding: 0;
        border: 1px solid #dcdcdc;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        overflow: hidden;
        background-color: #ffffff; /* Add background color to the container itself */
    }

    .messages {
        flex-grow: 1; /* Takes up available space within the fixed height container */
        overflow-y: auto; /* Enables vertical scrolling when content exceeds the space */
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        /* background-color: #ffffff; Removed, now set on container */
    }

    .message {
        display: flex;
        margin-bottom: 0.5rem;
        max-width: 85%; /* Allow messages to be slightly wider */
    }

    .message.user {
        align-self: flex-end; /* Changed from justify-content for better alignment */
    }

    .message.bot {
        align-self: flex-start;
    }

    .message-content {
        /* padding: 0.8rem 1.2rem; */ /* Padding inherited from .messages now */
        border-radius: 1.2rem;
        background-color: #e9e9eb;
        color: #333;
        line-height: 1.4;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        padding: 0.8rem 1.2rem; /* Re-added padding here is fine too */
    }

    .message.user .message-content {
        background-color: #007bff; /* Changed from gradient to solid color */
        color: white;
        /* Keep user messages more distinctly rounded */
        border-bottom-right-radius: 0.5rem; /* Slightly flatten one corner */
    }

    .message.bot .message-content {
         /* Keep bot messages more distinctly rounded */
         border-bottom-left-radius: 0.5rem; /* Slightly flatten one corner */
    }

    .input-form {
        display: flex;
        gap: 0.75rem;
        padding: 1rem 1.5rem; /* Adjusted padding */
        background-color: #f9f9f9; /* Slightly different background for input area */
        border-top: 1px solid #e0e0e0;
        box-shadow: none; /* Removed shadow as container has one now */
    }

    input {
        flex-grow: 1;
        padding: 0.8rem 1rem; /* Adjusted padding */
        border: 1px solid #d1d1d1; /* Lighter border */
        border-radius: 0.8rem; /* More rounded */
        font-size: 1rem;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05); /* Subtle inset shadow */
    }
    input:focus {
        outline: none;
        border-color: #007bff; /* Highlight on focus */
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.2); /* Focus ring */
    }

    button {
        padding: 0.8rem 1.5rem; /* Adjusted padding */
        background-color: #007bff; /* Keep primary blue or choose a new one */
        color: white;
        border: none;
        border-radius: 0.8rem; /* More rounded */
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500; /* Slightly bolder text */
        transition: background-color 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
    }

    button:hover:not(:disabled) {
        background-color: #0056b3; /* Darker shade on hover */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Hover shadow */
    }

    button:disabled {
        background-color: #b0b0b0; /* Adjusted disabled color */
        cursor: not-allowed;
        opacity: 0.7;
    }

    .loading {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        /* Match bot message style */
        padding: 0.8rem 1.2rem;
        border-radius: 1.2rem;
        background-color: #e9e9eb;
        color: #333;
        line-height: 1.4;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        align-self: flex-start; /* Align loading indicator like bot message */
        border-bottom-left-radius: 0.5rem;
    }

    .dot {
        width: 8px;
        height: 8px;
        background-color: #666;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both; /* Added 'both' */
    }

    /* Keep bounce animation */
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1.0);
        }
    }
</style> 