const BACKEND_URL = "http://127.0.0.1:5000";

function login() {

    const user = document.getElementById("username").value.trim();
    const pass = document.getElementById("password").value.trim();

    if (!user || !pass) {
        document.getElementById("error").innerText = "Enter username and password";
        return;
    }

    fetch(`${BACKEND_URL}/login`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            username: user,
            password: pass
        })
    })
    .then(res => {
        if (!res.ok) throw new Error("Invalid Credentials");
        return res.json();
    })
    .then(data => {
        if (data.success) {
            window.location.href = "dashboard.html";
        } else {
            document.getElementById("error").innerText = "Invalid Credentials";
        }
    })
    .catch(err => {
        document.getElementById("error").innerText = "Invalid Credentials";
    });
}