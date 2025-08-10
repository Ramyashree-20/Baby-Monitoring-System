// Function to handle chatbot interaction
async function sendMessage() {
    const userInput = document.getElementById('user-message');
    const message = userInput.value.trim();
    if (!message) return;

    const chatWindow = document.getElementById('chat-window');
    // Add user message
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'user-message';
    userMessageDiv.textContent = message;
    chatWindow.appendChild(userMessageDiv);
    userInput.value = '';
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Get bot response
    const response = await fetch('/ask_chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    });
    const data = await response.json();

    // Add bot message
    const botMessageDiv = document.createElement('div');
    botMessageDiv.className = 'bot-message';
    botMessageDiv.textContent = data.response;
    chatWindow.appendChild(botMessageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// NEW: Function to update the notifications list
async function updateNotifications() {
    const response = await fetch('/get_notifications');
    const notifications = await response.json();
    const listElement = document.getElementById('notifications-list');

    // Clear the current list
    listElement.innerHTML = '';

    if (notifications.length === 0) {
        listElement.innerHTML = '<p>No new notifications.</p>';
    } else {
        notifications.forEach(notif => {
            const p = document.createElement('p');
            p.textContent = notif;
            listElement.appendChild(p);
        });
    }
}

// Run the update function every 5 seconds (5000 milliseconds)
setInterval(updateNotifications, 5000);

// Also run it once on page load
document.addEventListener('DOMContentLoaded', updateNotifications);
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(registration => {
        console.log('Service Worker registered! Scope is: ', registration.scope);
      })
      .catch(err => {
        console.log('Service Worker registration failed: ', err);
      });
  });
}