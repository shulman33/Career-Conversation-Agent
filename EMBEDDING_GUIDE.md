# Embedding the Career Chatbot in Next.js

## Overview

This Gradio chatbot is configured for full-container embedding. It uses `fill_height=True` and `fill_width=True` to expand and fill its container completely, with no internal padding or max-width constraints.

## Gradio Server

The chatbot runs as a separate service and exposes a URL (e.g., `http://localhost:7860` in development). You will embed this via an iframe.

## Implementation

### Basic iframe Embed

```tsx
export default function ChatbotEmbed() {
  return (
    <iframe
      src="YOUR_GRADIO_URL"
      style={{
        width: '100%',
        height: '100%',
        border: 'none',
      }}
      allow="microphone"
    />
  );
}
```

### With Fixed Height Container

If embedding in a section with a fixed height:

```tsx
export default function ChatbotSection() {
  return (
    <div style={{ width: '100%', height: '600px' }}>
      <iframe
        src="YOUR_GRADIO_URL"
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
        }}
        allow="microphone"
      />
    </div>
  );
}
```

### Full-Page or Flex Container

For a chatbot that fills available space in a flex layout:

```tsx
export default function ChatbotPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Optional header */}
      <header style={{ padding: '1rem' }}>
        <h1>Chat with Sam</h1>
      </header>

      {/* Chatbot fills remaining space */}
      <div style={{ flex: 1, minHeight: 0 }}>
        <iframe
          src="YOUR_GRADIO_URL"
          style={{
            width: '100%',
            height: '100%',
            border: 'none',
          }}
          allow="microphone"
        />
      </div>
    </div>
  );
}
```

## Important Notes

1. **Parent container must have explicit dimensions** - The iframe will only fill space if its parent has a defined height (px, vh, %, or flex)

2. **`minHeight: 0`** - Required on flex children to allow proper shrinking; without this, content may overflow

3. **Environment variables** - Use `process.env.NEXT_PUBLIC_CHATBOT_URL` for the Gradio URL:
   ```tsx
   src={process.env.NEXT_PUBLIC_CHATBOT_URL}
   ```

4. **Loading state** - Consider adding a loading skeleton:
   ```tsx
   const [loaded, setLoaded] = useState(false);

   <iframe
     onLoad={() => setLoaded(true)}
     style={{ opacity: loaded ? 1 : 0 }}
     // ...
   />
   ```

5. **CORS** - Gradio handles CORS automatically, but if you encounter issues, ensure the Gradio server is launched with appropriate settings

## Tailwind CSS Example

```tsx
export default function ChatbotEmbed() {
  return (
    <div className="w-full h-[600px] lg:h-[800px]">
      <iframe
        src={process.env.NEXT_PUBLIC_CHATBOT_URL}
        className="w-full h-full border-0"
        allow="microphone"
        title="Career Chatbot"
      />
    </div>
  );
}
```

## Production Deployment

When deploying:

1. Deploy the Gradio app to a hosting service (Hugging Face Spaces, Railway, Render, etc.)
2. Set `NEXT_PUBLIC_CHATBOT_URL` in your Next.js environment to the production Gradio URL
3. Ensure HTTPS is used for both the Next.js app and Gradio server
