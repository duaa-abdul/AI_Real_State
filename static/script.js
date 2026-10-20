
const $ = (id) => document.getElementById(id);

const formatPKR = (amount) =>
    "PKR " + Number(amount).toLocaleString("en-PK");

async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.detail || "Request failed.");
    }

    return data;
}

function showError(element, message) {
    element.textContent = message;
    element.className = "result error-result";
}

function showSuccess(element, message) {
    element.textContent = message;
    element.className = "result success-result";
}

async function loadDashboard() {
    const stats = await apiRequest("/api/stats");

    $("totalProperties").textContent =
        stats.total_properties.toLocaleString();

    $("averagePrice").textContent = formatPKR(stats.average_price);
    $("minimumPrice").textContent = formatPKR(stats.minimum_price);
    $("averageArea").textContent =
        Math.round(stats.average_area).toLocaleString() + " sq ft";

    const chart = $("cityChart");
    chart.replaceChildren();

    const cities = Object.entries(stats.average_prices_by_city);
    const maxPrice = Math.max(...cities.map(([, price]) => price), 1);

    cities.forEach(([city, price]) => {
        const row = document.createElement("div");
        row.className = "bar-row";

        const label = document.createElement("span");
        label.textContent = city;

        const content = document.createElement("div");
        const track = document.createElement("div");
        track.className = "bar-track";

        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = `${(price / maxPrice) * 100}%`;
        bar.title = formatPKR(price);

        track.appendChild(bar);

        const value = document.createElement("div");
        value.className = "bar-value";
        value.textContent = formatPKR(price);

        content.append(track, value);
        row.append(label, content);
        chart.appendChild(row);
    });

    const properties = await apiRequest("/api/properties?limit=10");
    const table = $("propertyTable");
    table.replaceChildren();

    properties.forEach((property) => {
        const row = document.createElement("tr");

        [
            property.city,
            property.property_type,
            Number(property.area_sqft).toLocaleString(),
            property.bedrooms,
            formatPKR(property.price_pkr),
        ].forEach((value) => {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        });

        table.appendChild(row);
    });
}

$("predictionForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const result = $("predictionResult");
    result.textContent = "Calculating estimated price...";
    result.className = "result";

    const payload = {
        city: $("city").value,
        property_type: $("propertyType").value,
        area_sqft: Number($("area").value),
        bedrooms: Number($("bedrooms").value),
        bathrooms: Number($("bathrooms").value),
        age_years: Number($("age").value),
    };

    try {
        const data = await apiRequest("/api/predict", {
            method: "POST",
            body: JSON.stringify(payload),
        });

        showSuccess(
            result,
            `Estimated price: ${formatPKR(data.estimated_price_pkr)}. ` +
            data.disclaimer
        );
    } catch (error) {
        showError(result, error.message);
    }
});

function renderPropertyCards(properties, container) {
    container.replaceChildren();

    if (!properties.length) {
        container.textContent = "No matching properties found. Try changing your filters.";
        return;
    }

    properties.forEach((property) => {
        const card = document.createElement("article");
        card.className = "property-card";

        const title = document.createElement("h3");
        title.textContent = `${property.property_type} in ${property.city}`;

        const price = document.createElement("div");
        price.className = "price";
        price.textContent = formatPKR(property.price_pkr);

        const details = document.createElement("p");
        details.textContent =
            `${property.area_sqft.toLocaleString()} sq ft · ` +
            `${property.bedrooms} bedrooms · ${property.bathrooms} bathrooms`;

        const age = document.createElement("p");
        age.textContent = `Property age: ${property.age_years} years`;

        card.append(title, price, details, age);
        container.appendChild(card);
    });
}

$("recommendForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const container = $("recommendationResults");
    container.textContent = "Finding matching properties...";

    const bedroomValue = $("recBedrooms").value;

    const payload = {
        city: $("recCity").value || null,
        budget: Number($("budget").value),
        bedrooms: bedroomValue ? Number(bedroomValue) : null,
        property_type: $("recType").value || null,
        limit: 10,
    };

    try {
        const data = await apiRequest("/api/recommend", {
            method: "POST",
            body: JSON.stringify(payload),
        });

        renderPropertyCards(data.properties, container);
    } catch (error) {
        container.textContent = error.message;
    }
});

function addMessage(text, className) {
    const message = document.createElement("div");
    message.className = `message ${className}`;
    message.textContent = text;
    $("chatMessages").appendChild(message);
    message.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return message;
}

$("chatForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const input = $("question");
    const question = input.value.trim();

    if (!question) return;

    addMessage(question, "user-message");
    input.value = "";
    input.disabled = true;

    const button = $("chatForm").querySelector("button");
    button.disabled = true;
    button.textContent = "Thinking...";

    const answer = addMessage("Searching the knowledge base...", "ai-message");

    try {
        const data = await apiRequest("/api/chat", {
            method: "POST",
            body: JSON.stringify({ question }),
        });

        answer.textContent = data.answer +
            "\n\nRetrieved sources: " + data.sources.join(", ");
    } catch (error) {
        answer.textContent = "Assistant error: " + error.message;
    } finally {
        input.disabled = false;
        button.disabled = false;
        button.textContent = "Ask AI";
        input.focus();
    }
});

loadDashboard().catch((error) => {
    console.error("Dashboard loading failed:", error);
    $("propertyTable").textContent =
        "Unable to load properties. Check that the FastAPI server is running.";
});
