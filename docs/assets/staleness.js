
(function () {
  var WARN_HOURS = 36;
  var ESCALATE_HOURS = 72;
  var banner = document.getElementById("staleness-banner");
  var body = document.body;
  if (!banner || !body || !body.dataset.rebuiltAt) return;
  var rebuiltAt = new Date(body.dataset.rebuiltAt);
  if (isNaN(rebuiltAt.getTime())) return;
  var hours = (Date.now() - rebuiltAt.getTime()) / 3600000;
  if (hours >= ESCALATE_HOURS) {
    banner.textContent = "Data has not updated in over " + Math.floor(hours) + " hours. The automated pipeline may have stopped running.";
    banner.className = "staleness-banner staleness-critical";
    banner.hidden = false;
  } else if (hours >= WARN_HOURS) {
    banner.textContent = "Data was last updated " + Math.floor(hours) + " hours ago.";
    banner.className = "staleness-banner staleness-warning";
    banner.hidden = false;
  }
})();
