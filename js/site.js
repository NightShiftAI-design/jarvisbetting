const header = document.querySelector("[data-header]");
const nav = document.querySelector("[data-nav]");
const navToggle = document.querySelector("[data-nav-toggle]");
const revealItems = document.querySelectorAll(".reveal");
const inquiryForm = document.querySelector("[data-inquiry-form]");

const setHeaderState = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 8);
};

setHeaderState();
window.addEventListener("scroll", setHeaderState, { passive: true });

navToggle?.addEventListener("click", () => {
  const isOpen = nav?.classList.toggle("is-open");
  document.body.classList.toggle("nav-open", Boolean(isOpen));
  navToggle.setAttribute("aria-expanded", String(Boolean(isOpen)));
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("is-open");
    document.body.classList.remove("nav-open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.14 }
);

revealItems.forEach((item) => {
  const delay = item.dataset.delay;
  if (delay) item.style.setProperty("--delay", `${delay}ms`);
  revealObserver.observe(item);
});

inquiryForm?.addEventListener("submit", (event) => {
  event.preventDefault();

  const data = new FormData(inquiryForm);
  const status = inquiryForm.querySelector("[data-form-status]");
  const tourRequested = data.get("tour") ? "Yes" : "No";
  const lines = [
    "Celebration Event & Venue Inquiry",
    "",
    `Name: ${data.get("name") || ""}`,
    `Email: ${data.get("email") || ""}`,
    `Phone: ${data.get("phone") || ""}`,
    `Event type: ${data.get("eventType") || ""}`,
    `Preferred event date: ${data.get("date") || ""}`,
    `Estimated guest count: ${data.get("guests") || ""}`,
    `Tour requested: ${tourRequested}`,
    "",
    "Message:",
    data.get("message") || "",
  ];

  const subject = encodeURIComponent("Celebration Event & Venue inquiry");
  const body = encodeURIComponent(lines.join("\n"));
  window.location.href = `mailto:?subject=${subject}&body=${body}`;

  if (status) {
    status.textContent = "Your email app should open with the inquiry details. You can also call 865.900.8800.";
  }
});
