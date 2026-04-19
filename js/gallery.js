const filterButtons = document.querySelectorAll("[data-filter]");
const galleryItems = document.querySelectorAll(".gallery-item");
const lightbox = document.querySelector("[data-lightbox]");
const lightboxImage = document.querySelector("[data-lightbox-img]");
const lightboxClose = document.querySelector("[data-lightbox-close]");

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const filter = button.dataset.filter;
    filterButtons.forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");

    galleryItems.forEach((item) => {
      const categories = item.dataset.category || "";
      item.hidden = filter !== "all" && !categories.includes(filter);
    });
  });
});

galleryItems.forEach((item) => {
  item.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    const image = item.querySelector("img");
    lightboxImage.src = item.dataset.src;
    lightboxImage.alt = image?.alt || "Celebration Event and Venue gallery image";
    lightbox.hidden = false;
    document.body.classList.add("nav-open");
  });
});

const closeLightbox = () => {
  if (!lightbox || !lightboxImage) return;
  lightbox.hidden = true;
  lightboxImage.src = "";
  document.body.classList.remove("nav-open");
};

lightboxClose?.addEventListener("click", closeLightbox);
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLightbox();
});
