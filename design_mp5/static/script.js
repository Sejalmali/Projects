document.addEventListener('DOMContentLoaded', function() {
    const wrapper = document.querySelector('.wrapper');
    const loginLink = document.querySelector('.login-link'); 
    const registerLink = document.querySelector('.register-link');
    const btnPopup = document.querySelector('.btnLogin-popup');
    const iconClose = document.querySelector('.icon-close');
    // const registrationSuccess = "{{ session.get('registration_success', False) }}";
    // const loginSuccess = "{{ session.get('login_success', False) }}"; // Add this line

    // if (registrationSuccess) {
    //     alert('Registration successful!');
    // }

    // if (loginSuccess) { // Add this block
    //     alert('Welcome, you are in!');
    // }

    registerLink.addEventListener('click', () => {
        wrapper.classList.add('active');
    });

    loginLink.addEventListener('click', () => {
        wrapper.classList.remove('active');
    });

    btnPopup.addEventListener('click', () => {
        wrapper.classList.add('active-popup');
    });

    iconClose.addEventListener('click', () => {
        wrapper.classList.remove('active-popup');
    });
});
