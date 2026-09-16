const modal = document.getElementById("contactModal");
const openButton = document.getElementById("openContact");
const closeButton = document.getElementById("closeContact");
const backdrop = document.getElementById("closeContactBackdrop");
const form = document.getElementById("contactForm");

function openModal() {
  modal.hidden = false;
  document.body.style.overflow = "hidden";
  document.getElementById("name").focus();
}

function closeModal() {
  modal.hidden = true;
  document.body.style.overflow = "";
}

openButton.addEventListener("click", openModal);
closeButton.addEventListener("click", closeModal);
backdrop.addEventListener("click", closeModal);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modal.hidden) {
    closeModal();
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();

  const name = document.getElementById("name").value.trim();
  const email = document.getElementById("email").value.trim();
  const message = document.getElementById("message").value.trim();

  const subject = encodeURIComponent(`Anya Tennis Contact — ${name}`);
  const body = encodeURIComponent(
    `Name: ${name}\nEmail: ${email}\n\n${message}`
  );

  window.location.href =
    `mailto:smarttenniswithandy@gmail.com?subject=${subject}&body=${body}`;
});
