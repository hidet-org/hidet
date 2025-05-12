Release Guide
=============

This guide outlines the steps for releasing a new version of the **Hidet** project.
Hidet is actively developed and maintained by the Hidet team at **CentML Inc.**
This guide is intended for internal use, facilitating the periodic release process by upstreaming internal changes
to the public repository while integrating updates from the open-source project.

Sync Commits
------------

We should rebase internal commits onto the upstream ``main`` branch.
First, add remotes for both the internal and upstream repositories:

.. code-block:: bash

    git remote add upstream git@github.com:hidet-org/hidet
    git remote add origin <internal-repo>

Next, fetch the upstream repository and rebase internal commits on top of the upstream ``main`` branch:

.. code-block:: bash

    git checkout main                 # Switch to the internal repo's main branch
    git checkout -b release-x.y.z    # Create a new release branch
    git fetch upstream
    git rebase upstream/main         # Rebase internal commits on upstream main

.. note::

   During the rebase, you may encounter merge conflicts.
   Use your IDE (e.g., VSCode, PyCharm) to resolve them and continue the rebase until complete.

Submit a Pull Request to Upstream
---------------------------------

After rebasing, push the release branch to the upstream (public) repository:

.. code-block:: bash

    git push upstream release-x.y.z

Then, open a **pull request** (PR) from ``release-x.y.z`` to the ``main`` branch of the ``hidet-org/hidet`` repository for review.

Resolve Review Issues
----------------------

Address any issues raised during the review process.

* Do **not** introduce new features or substantial changes.
* Only make release-related adjustments or fixes needed to pass CI.

Merge the PR (Without Squashing)
--------------------------------

Once the PR is approved and CI passes:

1. Ensure that no new commits have been added to the ``main`` branch of the upstream repository.
   If there are new commits, rebase again before merging.
2. In the GitHub UI, **select the "Rebase and Merge" option**, **not** "Squash and merge" or "Merge commit".

.. warning::

   Preserving commit history is critical. Do not squash commits.

The final commit in this PR will act as the **cutoff commit** for the release.

Create the GitHub Release
--------------------------

Navigate to the **"Releases"** tab of the ``hidet-org/hidet`` repository and click **"Draft a new release"**.

* **Tag version**: Format as ``vx.y.z`` (e.g., ``v0.1.0``)
* **Release title**: Use ``Hidet x.y.z`` (e.g., ``Hidet 0.1.0``)
* **Release notes**: Modify the auto-generated notes as needed

For candidate or post releases, use:

* ``v0.1.0-rc1`` for release candidates
* ``v0.1.0.post1`` for post-release patches

Once the release is published, the CI pipeline will automatically build and upload the package to PyPI.

Final Verification
------------------

After publishing the release:

* ✅ Confirm that the release package is available on PyPI.
* ✅ Test that it installs and functions as expected.

Bump the Version Number
----------------------

After the release is published, update the version number in the codebase to the dev for next version.

.. code-block:: bash

   $ git fetch --all
   $ git checkout upstream/main
   $ git checkout -b bump-version
   $ python scripts/wheel/update_version.py --version x.y.z.dev
   $ git commit -m "bump version to x.y.z.dev"
   $ git push upstream bump-version

where ``x.y.z`` is the next hidet version. Create a PR to the upstream repository to merge the version bump.
After the PR is merged, reset the internal repository's main branch to points to the commit of upstream main branch.

.. code-block:: bash

   $ git checkout main
   $ git reset --hard upstream/main
   $ git push -f origin main
