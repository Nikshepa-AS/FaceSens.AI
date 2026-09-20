const API_URL = "http://127.0.0.1:8000";


async function loadUsers() {

    try {

        const response =
            await fetch(`${API_URL}/users`);

        const data =
            await response.json();

        document.getElementById(
            "userCount"
        ).textContent = data.count;

    } catch (error) {

        console.error(
            "Could not load users:",
            error
        );

    }
}


async function loadHistory() {

    const container =
        document.getElementById(
            "historyContainer"
        );


    try {

        const response =
            await fetch(`${API_URL}/history`);

        const data =
            await response.json();


        if (data.history.length === 0) {

            container.innerHTML = `
                <div class="empty-state">
                    No recognition activity yet.
                </div>
            `;

            return;
        }


        let html = "";


        data.history
            .slice(0, 5)
            .forEach(item => {

                const name =
                    item.name || "Unknown";

                const statusClass =
                    item.status === "known"
                        ? "known"
                        : "unknown";


                html += `
                    <div class="history-row">

                        <div>
                            <strong>
                                ${name}
                            </strong>

                            <small>
                                ${item.timestamp}
                            </small>
                        </div>

                        <span class="${statusClass}">
                            ${item.status.toUpperCase()}
                        </span>

                        <strong>
                            ${item.similarity.toFixed(3)}
                        </strong>

                    </div>
                `;

            });


        container.innerHTML = html;


    } catch (error) {

        container.innerHTML = `
            <div class="empty-state">
                Unable to load recognition history.
            </div>
        `;

        console.error(error);

    }
}


function startRecognition() {

    alert(
        "Live recognition module will open here."
    );
}


loadUsers();

loadHistory();