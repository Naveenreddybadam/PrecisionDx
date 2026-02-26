const BACKEND_URL = "http://127.0.0.1:5000";

document.addEventListener("DOMContentLoaded", () => {

    const loginForm = document.getElementById("loginForm");

    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            const username = document.getElementById("username").value;
            const password = document.getElementById("password").value;

            try {
                const res = await fetch(`${BACKEND_URL}/login`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username, password })
                });

                if (res.ok) {
                    window.location.href = "dashboard.html";
                } else {
                    alert("Invalid Credentials");
                }

            } catch {
                alert("Backend not running");
            }
        });
    }
});