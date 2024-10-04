// mapComponent.js
export function initializeMap(containerId, center, zoom, markers) {
    const map = L.map(containerId).setView(center, zoom);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    markers.forEach(markerData => {
        if (markerData.options.icon) {
            // Create a marker with a custom HTML icon
            L.marker(markerData.position, {
                icon: L.divIcon(markerData.options.icon)
            }).addTo(map);
        } else {
            // Create a circle marker
            L.circleMarker(markerData.position, markerData.options).addTo(map);
        }
    });
}