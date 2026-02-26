const BACKEND_URL = "http://127.0.0.1:5000";

/* =========================
   ADD PATIENT
========================= */
function addPatient() {

    const name = document.getElementById("newName").value.trim();
    const age = document.getElementById("newAge").value.trim();
    const gender = document.getElementById("newGender").value;

    if (!name || !age || !gender) {
        alert("Fill all patient details");
        return;
    }

    fetch(`${BACKEND_URL}/add_patient`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            patient_name: name,
            age: age,
            gender: gender
        })
    })
    .then(res => res.json())
    .then(() => {
        alert("Patient Added Successfully (OP number auto-generated)");
        document.getElementById("newName").value = "";
        document.getElementById("newAge").value = "";
        document.getElementById("newGender").value = "";
        loadPatients();
    })
    .catch(() => alert("Error adding patient"));
}


/* =========================
   LOAD PATIENT LIST
========================= */
function loadPatients() {

    fetch(`${BACKEND_URL}/patients`)
    .then(res => res.json())
    .then(data => {

        const table = document.getElementById("patientTable");
        table.innerHTML = "";

        if (!data.length) {
            table.innerHTML = "<tr><td colspan='4'>No Patients Found</td></tr>";
            return;
        }

        data.forEach(p => {

            table.innerHTML += `
                <tr>
                    <td>OP-${p.id}</td>
                    <td>${p.patient_name}</td>
                    <td>${p.age}</td>
                    <td>${p.gender}</td>
                </tr>
            `;
        });

    })
    .catch(() => alert("Cannot load patients"));
}


/* =========================
   SEARCH BY OP NUMBER
========================= */
function filterTable() {

    const input = document.getElementById("searchInput").value.toLowerCase();
    const rows = document.querySelectorAll("#patientTable tr");

    rows.forEach(row => {
        const op = row.cells[0].innerText.toLowerCase();
        row.style.display = op.includes(input) ? "" : "none";
    });
}


/* =========================
   INIT
========================= */
window.onload = function() {
    loadPatients();
};