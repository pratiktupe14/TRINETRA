const express = require('express');
const axios = require('axios');
const app = express();
const PORT = 3001;

app.use(express.json());

// IP Validation helper
const isValidIP = (ip) => {
    const reg = /^(\d{1,3}\.){3}\d{1,3}$/;
    return reg.test(ip);
};

app.post('/webhook/security-event', async (req, res) => {
    const event = req.body;
    console.log(`[WEBHOOK] Received event:`, event);

    const { ip, failedAttempts, type } = event;

    // Validation
    if (!ip || !isValidIP(ip)) {
        return res.status(400).json({ error: "Invalid or missing IP address" });
    }
    if (typeof failedAttempts !== 'number') {
        return res.status(400).json({ error: "failedAttempts must be a number" });
    }

    // Immediate non-blocking response
    res.json({ message: "Event received and processing started" });

    // Asynchronously trigger the heavy Python multi-agent pipeline
    processEvent(event);
});

async function processEvent(event) {
    console.log(`[PIPELINE] Forwarding event to Python Multi-Agent system...`);
    try {
        // Forwarding to our existing FastAPI endpoint on port 8000
        const response = await axios.post('http://localhost:8000/webhook/security-event', {
            ip: event.ip,
            failedAttempts: event.failedAttempts,
            type: event.type,
            is_simulated: event.type === "SIMULATED_EVENT"
        });
        console.log(`[PIPELINE] Python system accepted event. Incident ID:`, response.data.incident_id);
    } catch (error) {
        console.error(`[PIPELINE] Error triggering Python pipeline:`, error.message);
    }
}

app.listen(PORT, () => {
    console.log(`Webhook server listening on http://localhost:${PORT}`);
});
